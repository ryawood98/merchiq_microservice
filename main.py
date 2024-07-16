import os
from datetime import timedelta
from datetime import datetime as dt
import pytz
import re
import json
import base64
from flask import Blueprint, render_template, request, Flask, flash, session, redirect, url_for, send_from_directory, Response, make_response, jsonify
import flask_login
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from wtforms import StringField, PasswordField, BooleanField, DecimalField, RadioField, SelectField, TextAreaField, FileField, EmailField
from werkzeug.security import generate_password_hash, check_password_hash
from wtforms.validators import DataRequired, InputRequired, Length, EqualTo, Email, AnyOf
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_sqlalchemy import SQLAlchemy
from flask_talisman import Talisman
import psycopg2
import numpy as np
import pandas as pd
import stripe
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import sqlalchemy


################################################################################################
### initialize Flask app
################################################################################################

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')

# configure Flask-Talisman (for forcing HTTPS)
csp = {
    'default-src': '\'self\'',
    'script-src': '\'self\'',
}
talisman = Talisman(app,
    content_security_policy=False,#csp,
    content_security_policy_nonce_in=['script-src', 'style-src']
)

# define Flask-Limiter object
limiter = Limiter(key_func=get_remote_address)

# configure Flask-Session
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)

# configure Flask-Limiter
app.config['RATELIMIT_DEFAULT'] = '1000 / day, 500 / hour'
app.config['RATELIMIT_STORAGE_URI'] = 'memory://'
limiter.init_app(app)
@app.errorhandler(429)
def ratelimit_handler(e):
  return

# configure Flask-WTF
csrf = CSRFProtect(app)




################################################################################################
### create endpoint
################################################################################################

@app.route("/")
def index():
    return "Hello this is the new version!"



if __name__=='__main__':
    app.run()