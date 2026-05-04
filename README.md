# F1 Race Outcome Predictor

> Predict Formula 1 race finishing positions using 5 seasons of real telemetry data.

[Live Demo](https://f1-race-predictor-bl8o.onrender.com)**

---

##✨Features

- **Single Prediction** — Select a driver, team, circuit and grid position to predict their finishing position with probability bars
- **Head to Head** — Compare two drivers side by side and see who the model thinks will finish ahead
- **Season Stats** — Full driver leaderboard ranked by average historical finishing position with a bar chart
- **Circuit Filter** — Each circuit shows historically strong drivers based on past performance at that track
- **Real Data** — Trained on 5 seasons (2021–2025) of official F1 telemetry fetched via the FastF1 API

---

## How It Works

```
FastF1 API → Raw telemetry data → pandas cleaning → 
Random Forest model → Flask API → Web dashboard
```

1. **Data Collection** — Race results fetched from the official F1 timing API via FastF1 across 2021–2025 seasons (~1778 race entries, 29 circuits)
2. **Feature Engineering** — Driver, team, circuit, and grid position encoded as features
3. **Model** — Random Forest Classifier (200 estimators) trained on historical finishing positions
4. **Metrics** — Mean Absolute Error of **3.62 positions**, exact accuracy of **17.7%**
5. **Deployment** — Flask app deployed on Render with automated model retraining on startup

---

## Tech Stack


Data->FastF1, pandas
Model->scikit-learn (Random Forest)
Backend->Python, Flask
Frontend->HTML, CSS, Chart.js
Deployment->Render

---

## Model Performance

Mean Absolute Error->3.62 positions
Exact Accuracy->17.7%
Training Data->1,778 race entries
Seasons->2021–2025
Circuits->29

> **Note:** F1 is inherently unpredictable—safety cars, mechanical failures, and weather make perfect prediction impossible. An MAE of 3.62 is significantly better than a random baseline (~6-7 positions).

---

## Future Improvements

- [ ] Add weather data as a feature
- [ ] Include qualifying lap time delta vs teammate
- [ ] Tyre strategy prediction
- [ ] React frontend rewrite
- [ ] Auto-retrain after each new race weekend

---

## Author

**Bhavya Sharma**  
[GitHub](https://github.com/Bhavyash2105) · [LinkedIn](https://www.linkedin.com/in/bhavya-sharma-78212a333/)

---

<p align="center">Made with ❤️ and way too much F1 data</p>
