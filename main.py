import os
from datetime import datetime as dt
from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd
import sqlalchemy
from flask import Flask, jsonify, request
from pydantic import BaseModel, ValidationError
from flask_cors import CORS

# Create the flask app
app = Flask(__name__)
CORS(app)

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
        error = str(e)
        res = PredictionResponse(
            status=error,
            data=None,
        )
        return jsonify(res.model_dump()), 400

    min_price = req.min_price
    max_price = req.max_price
    item_names = req.item_names
    retailer = req.retailer

    # Query the database to get the data that we need
    query = "SELECT * FROM promos WHERE item_name IN :item_names \
AND non_promo_price BETWEEN :min_price AND :max_price \
AND retailer = :retailer;"
    with engine.begin() as conn:
        df = pd.read_sql(
            sqlalchemy.text(query),
            conn,
            params={
                "item_names": tuple(item_names),
                "min_price": min_price,
                "max_price": max_price,
                "retailer": retailer,
            },
            parse_dates=["retailer_week"],
        )

    # TPR = Temporary Price Reduction
    df["tpr_disc"] = df["non_promo_price"] - df["promo_price"].fillna(df["non_promo_price"])
    df["is_offer"] = df["offer_message_0"].notnull().astype(int)
    if df.shape[0] == 0:
        res = PredictionResponse(
            status="NO ITEMS FOUND",
            data=None,
        )
        return jsonify(res.model_dump()), 400

    # Look at the minimum date in our data but synced with the cadence
    # of the minimum and maximum promotion date
    min_df_date = df["retailer_week"].min()
    right_now = dt.now()
    two_years_ago = right_now - timedelta(days=7 * 52 * 2)
    one_year_from_now = right_now + timedelta(days=7 * 52)

    min_date = min_df_date - timedelta(
        days=((min_df_date - two_years_ago).days // 7) * 7
    )
    max_date = min_df_date + timedelta(
        days=((one_year_from_now - min_df_date).days // 7) * 7
    )
    calendar = pd.DataFrame(
        columns=["retailer_week"],
        data=np.arange(min_date, max_date, timedelta(days=7)),
    )
    calendar["week"] = calendar["retailer_week"].dt.isocalendar().week
    calendar["year"] = calendar["retailer_week"].dt.isocalendar().year

    # Create prediction for adblock
    ## Create merch grid
    mechanic_col = "tpr_disc"
    merch_grid_raw = (
        df.loc[df["non_promo_price"].between(min_price, max_price)]
        .groupby(by=["retailer_week"])[[mechanic_col]]
        .median()
        .dropna()
    )
    merch_grid_raw = calendar.merge(merch_grid_raw, on="retailer_week", how="left")
    merch_grid_raw[mechanic_col] = merch_grid_raw[mechanic_col].fillna(0.0).round(2)

    # Create discount model
    ## Create lagged promo
    merch_grid_raw[mechanic_col + "_ly"] = (
        merch_grid_raw[mechanic_col].shift(52).fillna(0.0)
    )

    # Create smoothed ly promo
    merch_grid_raw[mechanic_col + "_smoothed"] = (
        merch_grid_raw[mechanic_col + "_ly"]
        .rolling(window=3, center=True, win_type="gaussian", min_periods=1)
        .mean(std=1)
    )
    merch_grid_raw[mechanic_col + "_predicted"] = merch_grid_raw[
        mechanic_col + "_smoothed"
    ]

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
        status="GOOD REQUEST",
        data=data_pred,
    )
    return jsonify(res.model_dump()), 200


if __name__ == "__main__":
    app.run()
