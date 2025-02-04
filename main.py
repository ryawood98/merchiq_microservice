import os
import re
from datetime import datetime as dt
from datetime import timedelta
from typing import List, Literal

import numpy as np
import pandas as pd
import sqlalchemy
from flask import Flask, jsonify, request
from flask_cors import CORS
from pydantic import BaseModel, ValidationError

from helpers import generate_promo_plan


# Create the flask app
app = Flask(__name__)
CORS(app)

# Connect to the database
engine = sqlalchemy.create_engine(os.environ.get("DATABASE_URL"))


class PredictionRequest(BaseModel):
    min_price: float
    max_price: float
    item_names: List[str]
    retailers: List[str]
    use_entire_brand: bool = True


class DataSeries(BaseModel):
    week: List[dt]
    discount: List[float]
    label: List[str]


class CsvData(BaseModel):
    csv_payload: str


class DataPred(BaseModel):
    historical_tpr: DataSeries
    prediction_tpr: DataSeries
    historical_crl: DataSeries
    prediction_crl: DataSeries
    historical_coupon: DataSeries
    prediction_coupon: DataSeries
    download_content: CsvData


class PredictionResponse(BaseModel):
    status: str
    data: DataPred | None


class CalendarRequest(BaseModel):
    promo_type: Literal["tpr", "crl", "coupon", "weekly ad"]
    retailer: str
    brand: str


class CalendarResponse(BaseModel):
    status: str
    data: List[dict] | None


class GenerateRequest(BaseModel):
    promo_type: Literal["tpr", "crl", "coupon", "weekly ad"]
    retailer: str
    competitor_brand: str
    own_brand: str
    # Add any additional parameters needed for generation


class GenerateResponse(BaseModel):
    status: str
    data: List[dict] | None




def build_promo_list(retailer, brand, promo_type):
    query = """
        SELECT retailer_week, item_id, tpr_disc_unitized, tpr_disc_unitized_desc,
            crl_disc_unitized, crl_disc_unitized_desc, coupon_quantity_threshold, coupon_disc_unitized, coupon_disc_unitized_desc,
            (CASE WHEN promo_type='weekly ad' THEN 'Weekly Ad Feature' ELSE NULL END) AS weekly_ad
        FROM promos 
        WHERE retailer = :retailer 
        AND brand = :brand
        ORDER BY retailer_week;
    """

    with engine.begin() as conn:
        df = pd.read_sql(
            sqlalchemy.text(query),
            conn,
            params={
                "retailer": retailer,
                "brand": brand,
            },
            parse_dates=["retailer_week"],
        )

    if df.empty:
        raise Exception("No promotions found for the given criteria")

    # bucket TPR and coupon discounts for better grouping
    # TODO: handle percent TPRs and non-quantity-threshold=1 coupons
    def bucket_discounts(disc, suffix):
        if pd.isnull(disc):
            return None
        elif disc<0.5:
            return "$0.00-$0.50 " + suffix
        elif disc<=1:
            return "$0.50-$1.00 " + suffix
        elif disc<=2:
            return "$1-$2 " + suffix
        elif disc>=3:
            return "$2-$3 " + suffix
        elif disc<=5:
            return "$3-$5 " + suffix
        elif disc<=10:
            "$5-10 " + suffix
        elif disc>10:
            return "$10+ " + suffix
        else:
            return None
    df["tpr_disc_unitized_desc_bucketed"] = None
    df.loc[(df["tpr_disc_unitized"].notnull()),"tpr_disc_unitized_desc_bucketed"] = df.loc[(df["tpr_disc_unitized"].notnull()),"tpr_disc_unitized"].apply(bucket_discounts, suffix="TPR")
    df["coupon_disc_unitized_desc_bucketed"] = None
    df.loc[(df["coupon_quantity_threshold"]==1), "coupon_disc_unitized_desc_bucketed"] = df.loc[(df["coupon_quantity_threshold"]==1), "coupon_disc_unitized"].apply(bucket_discounts, suffix="Coupon")

    # Clean and prepare item IDs
    df["item_id"] = (
        df["item_id"]
        .dropna()
        .apply(
            lambda x: re.match(r"{?(\d+\-?\d*\-?\d*)(?:\.0)?}?", x)
            .group(1)
            .replace("-", "")
        )
        .astype(int)
    )

    # Group columns for aggregation
    cols_groups = {
        "tpr": [
            "tpr_disc_unitized_desc_bucketed",
        ],
        "crl": [
            "crl_disc_unitized_desc",
        ],
        "coupon": [
            "coupon_disc_unitized_desc_bucketed",
        ],
        "weekly ad": ["weekly_ad"],
    }

    # Aggregate promotions
    promos = (
        df.groupby(by=cols_groups[promo_type], dropna=False)
        .apply(
            lambda x: pd.Series(
                {
                    "retailer_week": x["retailer_week"].dropna().unique(),
                    "item_id": x["item_id"].dropna().unique(),
                }
            ), include_groups=False
        )
        .reset_index()
    )
    promos = promos.dropna(subset=cols_groups[promo_type], how="all")

    promos["retailer_week"] = promos["retailer_week"].apply(build_date_groups)
    promos = promos.explode("retailer_week")
    promos["promo_start_date"] = promos["retailer_week"].apply(lambda x: x[0])

    # Create predictions
    promo_pred = promos.loc[
        (promos["promo_start_date"] > (dt.now() - timedelta(days=364)))
    ].copy()

    promo_pred["promo_start_date"] += timedelta(days=364)
    promo_pred["retailer_week"] = promo_pred["retailer_week"].apply(
        lambda x: [d + timedelta(days=365) for d in x]
    )
    promo_pred["promo_length_weeks"] = promo_pred["retailer_week"].apply(len)
    promo_pred["promo_end_date"] = (
        (
            promo_pred["promo_start_date"]
            + timedelta(days=7) * promo_pred["promo_length_weeks"]
            - timedelta(days=1)
        )
        if not promo_pred.empty
        else np.nan
    )

    np.random.seed(42)
    promo_pred["probability"] = np.random.beta(7, 3, size=promo_pred.shape[0])

    # create display string
    if promo_type == "tpr":
        promo_pred["display_string"] = promo_pred["tpr_disc_unitized_desc_bucketed"]
    elif promo_type == "crl":
        promo_pred["display_string"] = promo_pred["crl_disc_unitized_desc"]
    elif promo_type == "coupon":
        promo_pred["display_string"] = promo_pred["coupon_disc_unitized_desc_bucketed"]
    elif promo_type == "weekly ad":
        promo_pred["display_string"] = "Weekly Ad Feature"

    return promo_pred


def convert_numpy_arrays(obj):
    """Recursively convert numpy arrays to lists in a nested structure."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_arrays(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_arrays(item) for item in obj]
    return obj


def build_date_groups(arr):
    """Group consecutive dates that are 7 days apart."""
    arr = sorted(arr)
    date_groups = [[arr[0]]]
    for d in arr[1:]:
        if (d - date_groups[-1][-1]).days == 7:
            date_groups[-1].append(d)
        else:
            date_groups.append([d])
    return date_groups


def replace_nan_with_none(obj):
    """Recursively replace numpy NaN values with None in a nested structure."""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    elif isinstance(obj, dict):
        return {key: replace_nan_with_none(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan_with_none(item) for item in obj]
    return obj


@app.route("/prediction", methods=["POST"])
def prediction():
    """Handle prediction requests for promotional data."""
    try:
        req = PredictionRequest(**request.json)
    except ValidationError as e:
        print(e)
        error = str(e)
        res = PredictionResponse(
            status=error,
            data=None,
        )
        return jsonify(res.model_dump()), 400

    min_price = req.min_price
    max_price = req.max_price
    item_names = req.item_names
    retailers = req.retailers
    use_entire_brand = req.use_entire_brand# defaults to False based on PredictionRequest class

    query = """
        SELECT DISTINCT retailer_week,non_promo_price_est,promo_price,tpr_disc_unitized,tpr_disc_unitized_desc,
            crl_disc_unitized,crl_disc_unitized_desc,
            coupon_disc_unitized,coupon_disc_unitized_desc,
            offer_message_0, offer_message_1, offer_message_2 FROM promos 
        WHERE {}
        AND (non_promo_price_est IS NULL 
             OR non_promo_price_est BETWEEN :min_price AND :max_price) 
        AND retailer IN :retailers;
    """.format("brand IN (SELECT DISTINCT brand FROM promos WHERE item_name IN :item_names)" if use_entire_brand 
              else "item_name IN :item_names")

    with engine.begin() as conn:
        df = pd.read_sql(
            sqlalchemy.text(query),
            conn,
            params={
                "item_names": tuple(item_names),
                "min_price": min_price,
                "max_price": max_price,
                "retailers": tuple(retailers),
            },
            parse_dates=["retailer_week"],
        )

    if df.shape[0] == 0:
        print("No items found")

    # print("query result df[['retailer_week', 'coupon_disc_unitized_desc']]: ", df.iloc[:50][['retailer_week', 'coupon_disc_unitized_desc']])

    ####### Calendar Template ############

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

    historical_week = calendar.loc[
        (calendar["retailer_week"] < dt.now()),
        "retailer_week",
    ].to_list()
    prediction_week = calendar.loc[
        (calendar["retailer_week"] >= dt.now()),
        "retailer_week",
    ].to_list()

    merch_grid_master = calendar.copy()

    ####### TPR Modeling ############

    # Create prediction for adblock
    ## Create merch grid
    merch_grid = (
        df.groupby(by=["retailer_week"])[
            ["tpr_disc_unitized", "tpr_disc_unitized_desc"]
        ]
        .apply(
            lambda x: (
                x.loc[
                    x["tpr_disc_unitized"].idxmax(),
                    ["tpr_disc_unitized", "tpr_disc_unitized_desc"],
                ]
                if x.dropna().shape[0] > 0
                else pd.Series(
                    {"tpr_disc_unitized": np.nan, "tpr_disc_unitized_desc": "$0.00"}
                )
            )
        )
        .rename(
            {
                "tpr_disc_unitized": "tpr_disc",
                "tpr_disc_unitized_desc": "tpr_disc_desc",
            },
            axis=1,
        )
    )
    merch_grid = calendar.merge(merch_grid, on="retailer_week", how="left")
    merch_grid["tpr_disc"] = merch_grid["tpr_disc"].fillna(0.0).round(2)
    merch_grid["tpr_disc_desc"] = merch_grid["tpr_disc_desc"].fillna("$0.00")

    # Create lagged promo
    merch_grid["tpr_disc_ly"] = merch_grid["tpr_disc"].shift(52).fillna(0.0)
    merch_grid["tpr_disc_desc_ly"] = merch_grid["tpr_disc_desc"].shift(52)

    # Create smoothed ly promo
    merch_grid["tpr_disc_smoothed"] = (
        merch_grid["tpr_disc_ly"]
        .rolling(window=3, center=True, win_type="gaussian", min_periods=1)
        .mean(std=1)
    )
    merch_grid["tpr_disc_predicted"] = merch_grid["tpr_disc_smoothed"]
    merch_grid["tpr_disc_desc_predicted"] = "$" + merch_grid[
        "tpr_disc_predicted"
    ].apply(lambda x: "{:.2f}".format(x)) + " Discount"

    # Merge TPR data into master merch grid
    merch_grid_master = merch_grid_master.merge(
        merch_grid, on=["retailer_week", "week", "year"]
    )

    # Create lists to send to client
    historical_disc_tpr = merch_grid.loc[
        (merch_grid["retailer_week"] < dt.now()), "tpr_disc"
    ].to_list()
    historical_disc_tpr_desc = merch_grid.loc[
        (merch_grid["retailer_week"] < dt.now()), "tpr_disc_desc"
    ].to_list()
    prediction_disc_tpr = merch_grid.loc[
        (merch_grid["retailer_week"] >= dt.now()), "tpr_disc_predicted"
    ].to_list()
    prediction_disc_tpr_desc = merch_grid.loc[
        (merch_grid["retailer_week"] >= dt.now()), "tpr_disc_desc_predicted"
    ].to_list()

    ####### CRL Modeling ############

    # Create prediction for adblock
    ## Create merch grid
    merch_grid = (
        df.groupby(by=["retailer_week"])[
            ["crl_disc_unitized", "crl_disc_unitized_desc"]
        ]
        .apply(
            lambda x: (
                x.loc[
                    x["crl_disc_unitized"].idxmax(),
                    ["crl_disc_unitized", "crl_disc_unitized_desc"],
                ]
                if x.dropna().shape[0] > 0
                else pd.Series(
                    {"crl_disc_unitized": np.nan, "crl_disc_unitized_desc": "$0.00"}
                )
            )
        )
        .rename(
            {
                "crl_disc_unitized": "crl_disc",
                "crl_disc_unitized_desc": "crl_disc_desc",
            },
            axis=1,
        )
    )
    merch_grid = calendar.merge(merch_grid, on="retailer_week", how="left")
    merch_grid["crl_disc"] = merch_grid["crl_disc"].fillna(0.0).round(2)
    merch_grid["crl_disc_desc"] = merch_grid["crl_disc_desc"].fillna("$0.00")

    # Create discount model
    ## Create lagged promo
    merch_grid["crl_disc_ly"] = merch_grid["crl_disc"].shift(52).fillna(0.0)
    merch_grid["crl_disc_desc_ly"] = merch_grid["crl_disc_desc"].shift(52)

    # Create smoothed ly promo
    merch_grid["crl_disc_smoothed"] = (
        merch_grid["crl_disc_ly"]
        .rolling(window=3, center=True, win_type="gaussian", min_periods=1)
        .mean(std=1)
    )
    merch_grid["crl_disc_predicted"] = merch_grid["crl_disc_smoothed"]
    merch_grid["crl_disc_desc_predicted"] = merch_grid["crl_disc_desc_ly"]

    # Merge coupon data into master merch grid
    merch_grid_master = merch_grid_master.merge(
        merch_grid, on=["retailer_week", "week", "year"]
    )

    # Create lists to send to client
    historical_disc_crl = merch_grid.loc[
        (merch_grid["retailer_week"] < dt.now()), "crl_disc"
    ].to_list()
    historical_disc_crl_desc = merch_grid.loc[
        (merch_grid["retailer_week"] < dt.now()), "crl_disc_desc"
    ].to_list()
    prediction_disc_crl = merch_grid.loc[
        (merch_grid["retailer_week"] >= dt.now()),
        "crl_disc_predicted",
    ].to_list()
    prediction_disc_crl_desc = merch_grid.loc[
        (merch_grid["retailer_week"] >= dt.now()),
        "crl_disc_desc_predicted",
    ].to_list()

    ####### Coupon Modeling ############

    # Create prediction for adblock
    ## Create merch grid
    merch_grid = (
        df.groupby(by=["retailer_week"])[
            ["coupon_disc_unitized", "coupon_disc_unitized_desc"]
        ]
        .apply(
            lambda x: (
                x.loc[
                    x["coupon_disc_unitized"].idxmax(),
                    ["coupon_disc_unitized", "coupon_disc_unitized_desc"],
                ]
                if x.dropna().shape[0] > 0
                else pd.Series(
                    {
                        "coupon_disc_unitized": np.nan,
                        "coupon_disc_unitized_desc": "$0.00",
                    }
                )
            )
        )
        .rename(
            {
                "coupon_disc_unitized": "coupon_disc",
                "coupon_disc_unitized_desc": "coupon_disc_desc",
            },
            axis=1,
        )
    )
    merch_grid = calendar.merge(merch_grid, on="retailer_week", how="left")
    merch_grid["coupon_disc"] = merch_grid["coupon_disc"].fillna(0.0).round(2)
    merch_grid["coupon_disc_desc"] = merch_grid["coupon_disc_desc"].fillna("$0.00")

    # Create discount model
    ## Create lagged promo
    merch_grid["coupon_disc_ly"] = merch_grid["coupon_disc"].shift(52).fillna(0.0)
    merch_grid["coupon_disc_desc_ly"] = merch_grid["coupon_disc_desc"].shift(52)

    # Create smoothed ly promo
    merch_grid["coupon_disc_smoothed"] = (
        merch_grid["coupon_disc_ly"]
        .rolling(window=3, center=True, win_type="gaussian", min_periods=1)
        .mean(std=1)
    )
    merch_grid["coupon_disc_predicted"] = merch_grid["coupon_disc_smoothed"]
    merch_grid["coupon_disc_desc_predicted"] = merch_grid["coupon_disc_desc_ly"]

    # Merge coupon data into master merch grid
    merch_grid_master = merch_grid_master.merge(
        merch_grid, on=["retailer_week", "week", "year"]
    )

    # Create lists to send to client
    historical_disc_coupon = merch_grid.loc[
        (merch_grid["retailer_week"] < dt.now()), "coupon_disc"
    ].to_list()
    historical_disc_coupon_desc = merch_grid.loc[
        (merch_grid["retailer_week"] < dt.now()), "coupon_disc_desc"
    ].to_list()
    prediction_disc_coupon = merch_grid.loc[
        (merch_grid["retailer_week"] >= dt.now()), "coupon_disc_predicted"
    ].to_list()
    prediction_disc_coupon_desc = merch_grid.loc[
        (merch_grid["retailer_week"] >= dt.now()), "coupon_disc_desc_predicted"
    ].to_list()

    # create "Download" csv payload data to send to client
    csv_payload = merch_grid_master[
        [
            "retailer_week",
            "year",
            "week",
            "tpr_disc",
            "tpr_disc_desc",
            "tpr_disc_predicted",
            "crl_disc",
            "crl_disc_desc",
            "crl_disc_predicted",
            "coupon_disc",
            "coupon_disc_desc",
            "coupon_disc_predicted",
        ]
    ].to_csv(index=False)

    data_pred = DataPred(
        historical_tpr=DataSeries(
            week=historical_week,
            discount=historical_disc_tpr,
            label=historical_disc_tpr_desc,
        ),
        prediction_tpr=DataSeries(
            week=prediction_week,
            discount=prediction_disc_tpr,
            label=prediction_disc_tpr_desc,
        ),
        historical_crl=DataSeries(
            week=historical_week,
            discount=historical_disc_crl,
            label=historical_disc_crl_desc,
        ),
        prediction_crl=DataSeries(
            week=prediction_week,
            discount=prediction_disc_crl,
            label=prediction_disc_crl_desc,
        ),
        historical_coupon=DataSeries(
            week=historical_week,
            discount=historical_disc_coupon,
            label=historical_disc_coupon_desc,
        ),
        prediction_coupon=DataSeries(
            week=prediction_week,
            discount=prediction_disc_coupon,
            label=prediction_disc_coupon_desc,
        ),
        download_content=CsvData(csv_payload=csv_payload),
    )
    res = PredictionResponse(
        status="GOOD REQUEST",
        data=data_pred,
    )
    return jsonify(res.model_dump()), 200


@app.route("/calendar", methods=["POST"])
def calendar():
    """Handle calendar requests for promotional data."""
    try:
        req = CalendarRequest(**request.json)
    except ValidationError as e:
        print(e)
        error = str(e)
        res = CalendarResponse(
            status=error,
            data=None,
        )
        return jsonify(res.model_dump()), 400

    promo_type = req.promo_type
    retailer = req.retailer
    brand = req.brand

    # Build promo_pred dataframe
    try:
        promo_pred = build_promo_list(retailer, brand, promo_type)
    except Exception as e:
        res = CalendarResponse(
            status=str(e),
            data=None,
        )
        return jsonify(res.model_dump()), 400

    # Prepare response
    calendar_data = promo_pred.to_dict("records")
    for record in calendar_data:
        record["promo_start_date"] = record["promo_start_date"].isoformat()
        record["promo_end_date"] = record["promo_end_date"].isoformat()
        record["retailer_week"] = [x.isoformat() for x in record["retailer_week"]]

    res = CalendarResponse(
        status="GOOD REQUEST",
        data=calendar_data,
    )
    response_data = convert_numpy_arrays(res.model_dump())
    response_data = replace_nan_with_none(response_data)
    return jsonify(response_data), 200


@app.route("/generate", methods=["POST"])
def generate():
    """Handle generation requests for promotional data."""
    try:
        req = GenerateRequest(**request.json)
    except ValidationError as e:
        print(e)
        error = str(e)
        res = GenerateResponse(
            status=error,
            data=None,
        )
        return jsonify(res.model_dump()), 400

    promo_type = req.promo_type
    retailer = req.retailer
    competitor_brand = req.competitor_brand
    own_brand = req.own_brand

    # Build promo_pred dataframe
    try:
        promo_pred = build_promo_list(retailer, competitor_brand, promo_type)
    except Exception as e:
        res = GenerateResponse(
            status=str(e),
            data=None,
        )
        return jsonify(res.model_dump()), 400

    # Build promo_prev dataframe for own-brand previous year promotions
    try:
        promo_prev = build_promo_list(retailer, own_brand, promo_type)
    except Exception as e:
        res = GenerateResponse(
            status=str(e),
            data=None,
        )
        return jsonify(res.model_dump()), 400

    # Use OpenAI to generate new promotion calendar
    promo_pred["promo_type"] = promo_type
    promo_prev["promo_type"] = promo_type
    promo_generated_list = generate_promo_plan(promo_pred, promo_prev)

    # After processing, convert to calendar_data format
    for i in range(len(promo_generated_list)):
        promo_generated_list[i]["promo_start_date"] = promo_generated_list[i][
            "promo_start_date"
        ].isoformat()
        promo_generated_list[i]["promo_end_date"] = promo_generated_list[i][
            "promo_end_date"
        ].isoformat()
        promo_generated_list[i]["retailer_week"] = [
            x.isoformat() for x in promo_generated_list[i]["retailer_week"]
        ]

    res = GenerateResponse(
        status="GOOD REQUEST",
        data=promo_generated_list,
    )
    response_data = convert_numpy_arrays(res.model_dump())
    response_data = replace_nan_with_none(response_data)
    return jsonify(response_data), 200


if __name__ == "__main__":
    app.run()
