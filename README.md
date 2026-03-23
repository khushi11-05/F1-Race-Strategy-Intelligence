# 🏎️ F1 Race Strategy Intelligence

> An end-to-end data analytics and machine learning project built on Formula 1 telemetry data.
> Covers data engineering, exploratory analysis, statistical testing, ML modelling, and an interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square)
![FastF1](https://img.shields.io/badge/FastF1-Telemetry-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Live App](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen?style=flat-square)

---

## 🌐 Live Demo

👉 https://f1-race-strategy-intelligence-svkhu4gtoktcqfugob4743.streamlit.app/

---

## 📸 Dashboard Preview

*(Add your screenshot here for visual impact)*
<img width="959" height="440" alt="image" src="https://github.com/user-attachments/assets/3e93df20-6362-4bf8-b841-369c5f7c0d00" />


---

## ⭐ Key Highlights

* Built an end-to-end ML + analytics pipeline on F1 telemetry data
* Processed **3,000+ lap-level records** across 3 races
* Achieved **90% accuracy** using Logistic Regression (LOO-CV)
* Developed a **multi-page Streamlit dashboard** with 5 analytical modules
* Performed statistical testing (Spearman correlation, Kruskal-Wallis test)
* Delivered insights on race strategy, tire usage, and performance drivers

---

## 📌 Project Summary

This project ingests real F1 timing and telemetry data via the FastF1 API, performs
statistical analysis on race strategy patterns, trains a machine learning classifier
to predict podium finishes, and presents all findings through a live Streamlit dashboard.

| Metric               | Value                              |
| -------------------- | ---------------------------------- |
| Season covered       | 2023 (Rounds 1–3)                  |
| Races                | Bahrain · Saudi Arabia · Australia |
| Drivers              | 20                                 |
| Teams                | 10                                 |
| Total lap records    | 3,002                              |
| Clean racing laps    | 2,469                              |
| ML accuracy (LOO-CV) | 90.0%                              |
| Podium recall        | 88.9%                              |
| Podium F1-score      | 0.727                              |

---

## 🗂️ Project Structure

```
F1-Race-Strategy-Intelligence/
│
├── app.py                    # Streamlit dashboard
├── f1_data_collection.py     # Data ingestion via FastF1
├── f1_phase2_eda.ipynb       # Exploratory analysis
├── f1_phase3_ml.ipynb        # ML modelling
│
├── f1_data/                  # Generated datasets
│   ├── 2023_laps.csv
│   ├── 2023_pits.csv
│   ├── 2023_results.csv
│   └── 2023_weather.csv
│
└── requirements.txt
```

---

## 🔧 Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect data

```bash
python f1_data_collection.py
```

### 3. Run dashboard

```bash
streamlit run app.py
```

---

## 📊 Key Insights (EDA)

### Qualifying vs Finish

* Spearman correlation: **0.685 (strong relationship)**
* 83% of top-5 starters finished in top-5

### Tire Strategy

* HARD tires show fastest median lap times due to stable long stints
* Statistically significant difference across compounds (p ≈ 10⁻¹¹⁹)

### Pit Stops

* Most common strategy: **2 stops**
* Australia GP showed high variability due to race interruptions

### DNF Rate

* ~22% of drivers did not finish across 3 races

---

## 🤖 Machine Learning

### Problem

Predict whether a driver finishes on the podium (Top 3)

### Best Model

**Logistic Regression**

* Accuracy: 90%
* Recall: 88.9%
* F1 Score: 0.727

### Key Insight

👉 **Car performance (constructor strength) is a stronger predictor than starting position**

---

## 🎯 Business Impact

* Quantified the impact of strategy and reliability on race outcomes
* Demonstrated importance of constructor performance in podium prediction
* Built interpretable ML model for decision-support scenarios
* Delivered insights through an interactive analytics dashboard

---

## 📱 Dashboard Features

* 🏁 Season Overview (KPIs, standings, race outcomes)
* 🔢 Driver vs Driver comparison
* 🛞 Tire strategy insights
* 📉 Lap time and performance analysis
* 🤖 Podium prediction tool

---

## 🚀 Future Improvements

* Expand to multiple seasons (2018–2025)
* Add qualifying lap time as feature
* Implement advanced models (XGBoost)
* Improve class balance with top-5 prediction
* Enhance UI/UX with filters and interactivity

---

## 🛠️ Tech Stack

* **Data**: FastF1 API
* **Processing**: pandas, numpy
* **Statistics**: scipy
* **ML**: scikit-learn
* **Visualization**: matplotlib, seaborn, plotly
* **App**: Streamlit

---

## 👩‍💻 Author

Built as a portfolio project for a data analyst role.
Demonstrates: data engineering · EDA · statistical testing · machine learning · dashboarding · storytelling.
