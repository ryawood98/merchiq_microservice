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
    promo_type: Literal["offer", "coupon", "weekly ad"]
    retailer: str
    brand: str


class CalendarResponse(BaseModel):
    status: str
    data: List[dict] | None


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

    query = """
        SELECT * FROM promos 
        WHERE item_name IN :item_names 
        AND (non_promo_price IS NULL 
             OR non_promo_price BETWEEN :min_price AND :max_price) 
        AND retailer IN :retailers;
    """

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

    # TPR = Temporary Price Reduction
    idx_tpr_promo_price_null = df["tpr_disc"].isnull() & df["promo_price"].isnull()
    df.loc[idx_tpr_promo_price_null, "tpr_disc"] = 0
    idx_tpr_null = df["tpr_disc"].isnull()
    df.loc[idx_tpr_null, "tpr_disc"] = df.loc[idx_tpr_null, "non_promo_price"].fillna(
        (df.loc[idx_tpr_null, "promo_price"] / 0.8).round()
    ) - df.loc[idx_tpr_null, "promo_price"].fillna(
        df.loc[idx_tpr_null, "non_promo_price"]
    )
    df["is_offer"] = df["offer_message_0"].notnull().astype(int)
    if df.shape[0] == 0:
        print("No items found")

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
    def quantify_tpr(s):
        # given a series of products/metrics for one date, return a number and a displayable string for the tpr offer
        tpr_discount = s["tpr_disc"]
        tpr_discount_description = "$" + "{:.2f}".format(tpr_discount)
        return pd.Series(
            {"tpr_disc": tpr_discount, "tpr_disc_desc": tpr_discount_description}
        )

    merch_grid = df.apply(quantify_tpr, axis=1)
    merch_grid = df.drop(["tpr_disc"], axis=1).join(merch_grid, how="inner")
    merch_grid = (
        merch_grid.groupby(by=["retailer_week"])[["tpr_disc", "tpr_disc_desc"]]
        .max()
        .reset_index()
    )
    merch_grid = calendar.merge(merch_grid, on="retailer_week", how="left")
    merch_grid["tpr_disc"] = merch_grid["tpr_disc"].fillna(0.0).round(2)
    merch_grid["tpr_disc_desc"] = merch_grid["tpr_disc_desc"].fillna("$0.00")

    # Create discount model
    ## Create lagged promo
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
    ].apply(lambda x: "{:.2f}".format(x))

    # Merge coupon data into master merch grid
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
    def quantify_crl(s):
        ### given a series of products/metrics for one date, return a number and a displayable string for the crl offer

        if pd.notnull(s["quantity_threshold"]):
            if s["quantity_threshold"] > 1:

                # 2F10
                if pd.notnull(s["reward_total_price"]):
                    if pd.notnull(s["non_promo_price_est"]):
                        crl_disc = (
                            s["non_promo_price"]
                            - s["reward_total_price"] / s["quantity_threshold"]
                        )
                        return pd.Series(
                            {
                                "crl_disc": crl_disc,
                                "crl_disc_desc": "Buy "
                                + str(int(s["quantity_threshold"]))
                                + " Save $"
                                + "{:.2f}".format(crl_disc)
                                + " Each",
                            }
                        )
                    else:
                        return pd.Series({"crl_disc": 0.0, "crl_disc_desc": "$0.00"})

                # B5S5
                elif pd.notnull(s["reward_dollars"]):
                    crl_disc = s["reward_dollars"] / s["quantity_threshold"]
                    return pd.Series(
                        {
                            "crl_disc": crl_disc,
                            "crl_disc_desc": "Buy "
                            + str(int(s["quantity_threshold"]))
                            + " Save $"
                            + "{:.2f}".format(crl_disc)
                            + " Each",
                        }
                    )

                # BOGO
                elif pd.notnull(s["reward_percent"]):
                    if pd.notnull(s["non_promo_price_est"]):
                        crl_disc = (
                            s["non_promo_price_est"]
                            * s["reward_percent"]
                            / s["quantity_threshold"]
                        )
                        return pd.Series(
                            {
                                "crl_disc": crl_disc,
                                "crl_disc_desc": "Buy "
                                + str(int(s["quantity_threshold"]))
                                + " Save "
                                + str(int(s["reward_percent"]))
                                + "% of One",
                            }
                        )
                    else:
                        return pd.Series({"crl_disc": 0.0, "crl_disc_desc": "$0.00"})
            else:
                return pd.Series({"crl_disc": 0.0, "crl_disc_desc": "$0.00"})

        elif pd.notnull(s["spend_threshold"]):

            # S25S5
            if pd.notnull(s["reward_dollars"]):
                if pd.notnull(s["non_promo_price_est"]):
                    crl_disc = s["reward_dollars"] / (
                        s["spend_threshold"] / s["non_promo_price_est"]
                    )
                    return pd.Series(
                        {
                            "crl_disc": crl_disc,
                            "crl_disc_desc": "Spend $"
                            + "{:.2f}".format(s["spend_threshold"])
                            + " Save $"
                            + "{:.2f}".format(s["reward_dollars"]),
                        }
                    )
                else:
                    crl_disc = s["reward_dollars"]
                    return pd.Series(
                        {
                            "crl_disc": crl_disc,
                            "crl_disc_desc": "$" + "{:.2f}".format(crl_disc),
                        }
                    )
            else:
                return pd.Series({"crl_disc": 0.0, "crl_disc_desc": "$0.00"})

        else:
            return pd.Series({"crl_disc": 0.0, "crl_disc_desc": "$0.00"})

    merch_grid = df.apply(quantify_crl, axis=1)
    merch_grid = df.join(merch_grid, how="inner")
    merch_grid = (
        merch_grid.groupby(by=["retailer_week"])[["crl_disc", "crl_disc_desc"]]
        .max()
        .reset_index()
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
    def quantify_coupon(s):
        ### given a series of products/metrics for one date, return a number and a displayable string for the crl offer

        if pd.notnull(s["coupon_quantity_threshold"]):

            # $5 Off 3
            if pd.notnull(s["coupon_dollar_value"]):
                coupon_disc = s["coupon_dollar_value"] / s["coupon_quantity_threshold"]
                return pd.Series(
                    {
                        "coupon_disc": coupon_disc,
                        "coupon_disc_desc": "$"
                        + "{:.2f}".format(s["coupon_dollar_value"])
                        + " Off "
                        + str(int(s["coupon_quantity_threshold"])),
                    }
                )

        return pd.Series({"coupon_disc": 0.0, "coupon_disc_desc": "$0.00"})

    merch_grid = df.apply(quantify_coupon, axis=1)
    merch_grid = df.join(merch_grid, how="inner")
    merch_grid = (
        merch_grid.groupby(by=["retailer_week"])[["coupon_disc", "coupon_disc_desc"]]
        .max()
        .reset_index()
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

    query = """
        SELECT retailer_week, item_id, quantity_threshold, spend_threshold,
               reward_dollars, reward_percent, reward_total_price,
               coupon_dollar_value, coupon_percent_value, coupon_total_price,
               crl_string, coupon_string, promo_price_string,
               (CASE WHEN promo_type=\'weekly ad\' THEN \'Weekly Ad Feature\' ELSE NULL END) AS weekly_ad
        FROM promos 
        WHERE retailer = :retailer 
        AND brand ILIKE :brand
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
        res = CalendarResponse(
            status="No promotions found for the given criteria",
            data=None,
        )
        return jsonify(res.model_dump()), 404

    # Clean and prepare item IDs
    df["item_id"] = (
        df["item_id"]
        .dropna()
        .apply(
            lambda x: re.match(r"{(\d+\-?\d*\-?\d*)(?:\.0)?}", x)
            .group(1)
            .replace("-", "")
        )
        .astype(int)
    )

    # Group columns for aggregation
    cols_groups = {
    "offer":
        [
            "quantity_threshold",
            "spend_threshold",
            "reward_dollars",
            "reward_percent",
            "reward_total_price"
        ],
    "coupon":
        [
            "coupon_dollar_value",
            "coupon_percent_value",
            "coupon_total_price"
        ],
    "weekly ad":
        [
            "weekly_ad"
        ]
    }

    # Aggregate promotions
    promos = df.groupby(by=cols_groups[promo_type], dropna=False).apply(lambda x: pd.Series({
        'retailer_week': x['retailer_week'].dropna().unique(),
        'item_id': x['item_id'].dropna().unique(),
        'coupon_string': x['coupon_string'].dropna().unique(),
        'crl_string': x['crl_string'].dropna().unique(),
        'promo_price_string': x['promo_price_string'].dropna().unique()
    })).reset_index()
    promos = promos.dropna(subset=cols_groups[promo_type], how='all')

    promos["retailer_week"] = promos["retailer_week"].apply(build_date_groups)
    promos = promos.explode("retailer_week")
    promos["promo_start_date"] = promos["retailer_week"].apply(lambda x: x[0])

    # Create predictions
    promo_pred = promos.loc[
        (promos["promo_start_date"] > (dt.now() - timedelta(days=365)))
    ].copy()

    promo_pred["promo_start_date"] += timedelta(days=365)
    promo_pred["retailer_week"] = promo_pred["retailer_week"].apply(
        lambda x: [d + timedelta(days=365) for d in x]
    )
    promo_pred["promo_length_weeks"] = promo_pred["retailer_week"].apply(len)
    promo_pred["promo_end_date"] = (
        promo_pred["promo_start_date"]
        + timedelta(days=7) * promo_pred["promo_length_weeks"]
        - timedelta(days=1)
    ) if not promo_pred.empty else np.nan

    # Prepare response
    calendar_data = promo_pred.to_dict("records")
    for record in calendar_data:
        record["retailer_week"] = [x.isoformat() for x in record["retailer_week"]]

    res = CalendarResponse(
        status="GOOD REQUEST",
        data=calendar_data,
    )
    response_data = convert_numpy_arrays(res.model_dump())
    response_data = replace_nan_with_none(response_data)
    return jsonify(response_data), 200


if __name__ == "__main__":
    app.run()
