import os
from datetime import datetime as dt
from datetime import timedelta

import numpy as np
import pandas as pd
import sqlalchemy
from flask import Flask, jsonify, request

# Create the flask app
app = Flask(__name__)

# Connect to the database
engine = sqlalchemy.create_engine(os.environ.get("DATABASE_URL"))


@app.route("/prediction", methods=["GET"])
def prediction():
    if (
        not request.args.get("min_price")
        or not request.args.get("max_price")
        or not request.args.get("item_names")
        or not request.args.get("retailer")
    ):
        return jsonify({"status": "Missing parameters", "data": request.args}), 400
    try:
        min_price = float(request.args.get("min_price"))
        max_price = float(request.args.get("max_price"))
        item_names = request.args.getlist("item_names")
        retailer = request.args.get("retailer")
    except:
        return jsonify({"status": "Bad parameters"}), 400

    ### query database
    with engine.begin() as conn:
        df = pd.read_sql(
            sqlalchemy.text(
                "SELECT * FROM promos WHERE item_name IN :item_names AND non_promo_price BETWEEN :min_price AND :max_price AND retailer = :retailer;"
            ),
            conn,
            params={
                "item_names": tuple(item_names),
                "min_price": min_price,
                "max_price": max_price,
                "retailer": retailer,
            },
            parse_dates=["retailer_week"],
        )
    df["tpr_disc"] = df["non_promo_price"] - df["promo_price"]
    df["is_offer"] = df["offer_message_0"].notnull().astype(int)
    if df.shape[0] == 0:
        return jsonify({"status": "no items found"}), 400

    # create calendar to help create merch grid
    min_date = df["retailer_week"].min() - timedelta(
        days=7
        * (df["retailer_week"].min() - (dt.now() - timedelta(days=7 * 104))).days
        // 7
    )
    max_date = df["retailer_week"].min() + timedelta(
        days=7
        * (((dt.now() + timedelta(days=7 * 52)) - df["retailer_week"].min()).days // 7)
    )
    calendar = pd.DataFrame(
        columns=["retailer_week"], data=np.arange(min_date, max_date, timedelta(days=7))
    )
    calendar["week"] = calendar["retailer_week"].dt.isocalendar().week
    calendar["year"] = calendar["retailer_week"].dt.isocalendar().year
    years = sorted(calendar["year"].unique())

    ### create prediction for adblock

    # create merch grid
    mechanic_col = "tpr_disc"
    merch_grid_raw = (
        df.loc[(df["non_promo_price"].between(min_price, max_price))]
        .groupby(by=["retailer_week"])[[mechanic_col]]
        .first()
        .dropna()
    )
    merch_grid_raw = calendar.merge(merch_grid_raw, on="retailer_week", how="left")
    merch_grid_raw[mechanic_col] = merch_grid_raw[mechanic_col].fillna(0.0).round(2)
    merch_grid = merch_grid_raw.pivot_table(
        index="week",
        columns="year",
        values=mechanic_col,
        aggfunc=lambda x: next(iter(x.dropna()), None),
        dropna=False,
    )

    # add latest year's actual dates for reference
    min_date_latest_year = calendar.loc[
        (calendar["year"] == calendar["year"].max()), "retailer_week"
    ].min()
    latest_year_dates = pd.DataFrame(
        index=np.arange(52),
        data={
            "retailer_week": np.arange(
                min_date_latest_year,
                min_date_latest_year + timedelta(days=7 * 52),
                timedelta(days=7),
            )
        },
    )
    merch_grid = latest_year_dates.join(merch_grid)

    ### create discounting model

    # create lagged promo
    merch_grid_raw[mechanic_col + "_ly"] = (
        merch_grid_raw[mechanic_col].shift(52).fillna(0.0)
    )

    # create smoothed ly promo
    merch_grid_raw[mechanic_col + "_smoothed"] = (
        merch_grid_raw[mechanic_col + "_ly"]
        .rolling(window=3, center=True, win_type="gaussian", min_periods=1)
        .mean(std=1)
    )
    merch_grid_raw[mechanic_col + "_predicted"] = merch_grid_raw[
        mechanic_col + "_smoothed"
    ].shift(52)

    return (
        jsonify(
            {
                "status": "good request",
                "data": {
                    "historical": {
                        "week": merch_grid_raw.loc[
                            (merch_grid_raw["retailer_week"] < dt.now()),
                            "retailer_week",
                        ].to_list(),
                        "discount": merch_grid_raw.loc[
                            (merch_grid_raw["retailer_week"] < dt.now()), mechanic_col
                        ].to_list(),
                    },
                    "prediction": {
                        "week": merch_grid_raw.loc[
                            (merch_grid_raw["retailer_week"] >= dt.now()),
                            "retailer_week",
                        ].to_list(),
                        "discount": merch_grid_raw.loc[
                            (merch_grid_raw["retailer_week"] >= dt.now()),
                            mechanic_col + "_predicted",
                        ].to_list(),
                    },
                },
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run()
