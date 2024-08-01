import os
from datetime import datetime as dt
from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd
import sqlalchemy
from flask import Flask, jsonify, request
from pydantic import BaseModel, ValidationError

# Create the flask app
app = Flask(__name__)

# Connect to the database
engine = sqlalchemy.create_engine(os.environ.get("DATABASE_URL"))


class PredictionRequest(BaseModel):
    min_price: float
    max_price: float
    item_names: List[str]
    retailer: str


class DataSeries(BaseModel):
    week: List[dt]
    discount: List[float]


class DataPred(BaseModel):
    historical: DataSeries
    prediction: DataSeries


class PredictionResponse(BaseModel):
    status: str
    data: DataPred | None


@app.route("/prediction", methods=["POST"])
def prediction():
    try:
        req = PredictionRequest(**request.json)
    except ValidationError as e:
        error = e.errors()
        res = PredictionResponse(
            status=error,
            data=None,
        )
        return jsonify(res.model_dump()), 400

    min_price = req.min_price
    max_price = req.max_price
    item_names = req.item_names
    retailer = req.retailer

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
        res = PredictionResponse(
            status="No items found",
            data=None,
        )
        return jsonify(res.model_dump()), 400

    # create calendar to help create merch grid
    min_date = df["retailer_week"].min() - timedelta(
        days=7
        * ((df["retailer_week"].min() - (dt.now() - timedelta(days=7 * 104))).days // 7)
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

    historical_week = merch_grid_raw.loc[
        (merch_grid_raw["retailer_week"] < dt.now()),
        "retailer_week",
    ].to_list()
    historical_discount = merch_grid_raw.loc[
        (merch_grid_raw["retailer_week"] < dt.now()),
        mechanic_col,
    ].to_list()
    prediction_week = merch_grid_raw.loc[
        (merch_grid_raw["retailer_week"] >= dt.now()),
        "retailer_week",
    ].to_list()
    prediction_discount = merch_grid_raw.loc[
        (merch_grid_raw["retailer_week"] >= dt.now()),
        mechanic_col + "_predicted",
    ].to_list()

    data_pred = DataPred(
        historical=DataSeries(
            week=historical_week,
            discount=historical_discount,
        ),
        prediction=DataSeries(
            week=prediction_week,
            discount=prediction_discount,
        ),
    )
    res = PredictionResponse(
        status="Good request",
        data=data_pred,
    )
    return jsonify(res.model_dump()), 200


if __name__ == "__main__":
    app.run()
