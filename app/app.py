from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and encoders
with open("../model/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("../model/le_driver.pkl", "rb") as f:
    le_driver = pickle.load(f)
with open("../model/le_team.pkl", "rb") as f:
    le_team = pickle.load(f)

DRIVERS = list(le_driver.classes_)
TEAMS = list(le_team.classes_)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    selected_driver = None
    selected_team = None
    selected_grid = None

    if request.method == "POST":
        try:
            selected_driver = request.form["driver"]
            selected_team   = request.form["team"]
            selected_grid   = int(request.form["grid"])

            driver_enc = le_driver.transform([selected_driver])[0]
            team_enc   = le_team.transform([selected_team])[0]

            pred = model.predict([[selected_grid, driver_enc, team_enc]])[0]
            prediction = int(pred)
        except Exception as e:
            error = str(e)

    return render_template("index.html",
                           drivers=DRIVERS,
                           teams=TEAMS,
                           prediction=prediction,
                           error=error,
                           selected_driver=selected_driver,
                           selected_team=selected_team,
                           selected_grid=selected_grid)

if __name__ == "__main__":
    app.run(debug=True)