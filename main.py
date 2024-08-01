import os
from datetime import timedelta
from datetime import datetime as dt
from flask import (
    Blueprint,
    render_template,
    request,
    Flask,
    flash,
    session,
    redirect,
    url_for,
    send_from_directory,
    Response,
    make_response,
    jsonify,
)
import flask_login
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from wtforms import (
    StringField,
    PasswordField,
    BooleanField,
    DecimalField,
    RadioField,
    SelectField,
    TextAreaField,
    FileField,
    EmailField,
)
from werkzeug.security import generate_password_hash, check_password_hash
from wtforms.validators import (
    DataRequired,
    InputRequired,
    Length,
    EqualTo,
    Email,
    AnyOf,
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_sqlalchemy import SQLAlchemy
from flask_talisman import Talisman
import psycopg2
import numpy as np
import pandas as pd
import sqlalchemy
import sklearn.cluster


engine = sqlalchemy.create_engine(os.environ.get("DATABASE_URL"))


################################################################################################
### initialize Flask app
################################################################################################

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")

# configure Flask-Talisman (for forcing HTTPS)
csp = {
    "default-src": "'self'",
    "script-src": "'self'",
}
talisman = Talisman(
    app,
    content_security_policy=False,  # csp,
    content_security_policy_nonce_in=["script-src", "style-src"],
)

# define Flask-Limiter object
limiter = Limiter(key_func=get_remote_address)

# configure Flask-Session
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)

# configure Flask-Limiter
app.config["RATELIMIT_DEFAULT"] = "1000 / day, 500 / hour"
app.config["RATELIMIT_STORAGE_URI"] = "memory://"
limiter.init_app(app)


@app.errorhandler(429)
def ratelimit_handler(e):
    return


# configure Flask-WTF
# csrf = CSRFProtect(app)


################################################################################################
### create endpoint
################################################################################################


@app.route("/adblock", methods=("GET",))
def adblock():

    ### parse request arguments for list of search parameters
    retailer = request.args.get("retailer")
    sector = request.args.get("sector")
    category = request.args.get("category")
    brand = request.args.get("brand")
    item_name = request.args.get("item_name")
    # if retailer is None or sector is None or category is None or brand is None or item_name is None:
    #     return jsonify({'status':'invalid request'}), 400

    ### query database
    with engine.begin() as conn:
        df = pd.read_sql(
            sqlalchemy.text(
                "SELECT * FROM promos WHERE item_name ILIKE :item_name AND non_promo_price IS NOT NULL;"
            ),
            conn,
            params={"item_name": "%" + item_name + "%"},
        )
    df = (
        df.dropna(subset=["non_promo_price"])
        .drop_duplicates(subset=["non_promo_price"])
        .reset_index(drop=True)
    )

    ### calculate adblocks
    X = df["non_promo_price"].values.reshape(-1, 1)
    if X.shape[0] == 0:
        return jsonify({"error": "no items found"}), 404
    n_clusters_sums = np.zeros(5)
    for n_clusters in range(1, 6):
        clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters).fit(
            X
        )
        clusters = clustering.labels_
        n_clusters_sums[n_clusters - 1] = sum(
            [
                abs(X[clusters == i] - X[clusters == i].mean()).mean()
                for i in range(clustering.n_clusters_)
            ]
        )

    # determine best number of clusters by finding last n_clusters with big enough reduction in mean distance
    min_mean_distance = 1
    n_clusters = (
        np.where((n_clusters_sums[:-1] - n_clusters_sums[1:]) > min_mean_distance)[0][
            -1
        ]
        + 2
    )

    # define adblocks
    adblocks = dict()
    clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters).fit(X)
    for cluster_n in range(n_clusters):
        cluster_mean = X[clustering.labels_ == cluster_n].mean()
        cluster_sd = max(0.5, X[clustering.labels_ == cluster_n].std())
        adblocks[cluster_n] = {
            "min_price": max(0, round(cluster_mean - cluster_sd)),
            "max_price": round(cluster_mean + cluster_sd),
            "item_names": df["item_name"]
            .values[clustering.labels_ == cluster_n]
            .tolist(),
        }

    # re-order clusters
    keys = sorted(
        adblocks.keys(),
        key=lambda x: adblocks[x]["min_price"] + adblocks[x]["max_price"],
    )
    adblocks = {cluster_n: adblocks[keys[cluster_n]] for cluster_n in range(n_clusters)}

    ### return adblocks
    return jsonify(adblocks), 200


@app.route("/prediction", methods=("GET",))
def prediction():

    ### parse request arguments for list of adblocks
    # data = request.get_json()
    # print('data', data)
    if (
        not request.args.get("min_price")
        or not request.args.get("max_price")
        or not request.args.get("item_names")
        or not request.args.get("retailer")
    ):
        return jsonify({"status": "Missing parameters"}), 400
    try:
        min_price = float(request.args.get("min_price"))
        max_price = float(request.args.get("max_price"))
        item_names = request.args.getlist("item_names")
        print("item_names:", item_names)
    except:
        return jsonify({"status": "Bad parameters"}), 400

    ### query database
    with engine.begin() as conn:
        df = pd.read_sql(
            sqlalchemy.text(
                "SELECT * FROM promos WHERE item_name IN :item_names AND non_promo_price BETWEEN :min_price AND :max_price;"
            ),
            conn,
            params={
                "item_names": tuple(item_names),
                "min_price": min_price,
                "max_price": max_price,
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
