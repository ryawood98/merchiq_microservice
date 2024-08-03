# MerchIQ Microservices

## Project structure
The project is simple right now. The entire flask app is stored at `main.py`

## Dependency management
We accomplish this through `poetry` and will in the future deprecate the `requirements.txt` file. We accomplish this by using the global `python` package management software, known as `pipx`. To install this, select your version of `python>=3.9`, and perform the following commands:

```
python3 -m venv ~/.venvs/global
source ~/.venvs/global/bin/activate
python -m pip install --user pipx
echo "export PATH=~/.venvs/global/bin:$PATH" >> ~/.bashrc
source ~/.bashrc
deactivate

pipx install poetry
```

Afterwards, make sure you open another virtual environment for this repository:

```
python3 -m venv ~/.venvs/merchiq_microservice
source ~/.venvs/merchiq_microservice/bin/activate
poetry install
```

After the above is accomplished you should have all the packages necessary for this project. Run `pip freeze` to verify.








## Deploying app to Heroku
The app is deployed to Heroku by pushing the repo to Heroku. To add the heroku repository use the following command:

```
heroku git:remote -a merchiq-microservice
```

Then you can push changes with the following commands:

```
git add .
git commit -m "commit message"
git push heroku main
```

This requires you to push from a repo with access to the heroku project (currently ryawood98@gmail.com and kzhao16@gmail.com). It also may prompt you to login to heroku, in which case you can login with:

```
heroku login
```

which will open a browser window and allow you to log in to Heroku using 2-factor authentication.
