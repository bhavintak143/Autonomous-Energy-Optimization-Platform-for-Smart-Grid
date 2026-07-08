# ⚡ Autonomous Energy Optimization Platform for Smart Grid

An AI-powered dashboard that forecasts household energy demand, segments customers by usage pattern, and generates optimization insights — built on real London smart meter data.

**🔗 Live Demo:** [https://autonomous-energy-optimization-platform.streamlit.app/]

---

## 🚀 How to Run

Follow these steps in order:

1. **Clone or download this project** to your local machine.

2. **Install the required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook first**
   Open `main.ipynb` and execute all cells from top to bottom.
   This step trains the forecasting model and automatically creates an `outputs/` folder containing:
   - `energy_forecast_model.pkl`
   - `cleaned_merged_energy_data.csv.gz`
   - `household_usage_groups.csv`

   > ⚠️ The dashboard will not run without these files, so this step must be completed first.

4. **Run the dashboard**
   Once the `outputs/` folder has been generated, launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   By default, `app.py` looks for the `outputs/` folder alongside itself. If yours lives elsewhere, open the sidebar's **⚙️ Advanced: data source** panel and point it to the correct path.

```
Autonomous Energy Optimization Platform for Smart Grid/
├── main.ipynb
├── app.py
├── requirements.txt
├── README.md
└── outputs/
    ├── energy_forecast_model.pkl
    ├── cleaned_merged_energy_data.csv.gz
    └── household_usage_groups.csv
```

---

## 📊 Dashboard Features

- **Scenario Simulator** – adjust date, weather, and holiday inputs in the sidebar and get an instant energy prediction
- **Live Weather Forecast** – enter any city to pull a real 7-day forecast and project energy demand
- **Trend Explorer** – historical energy usage patterns over time, by ACORN group, and by temperature
- **Model Performance** – accuracy metrics (MAE, RMSE, R²), predicted-vs-actual plot, feature importance
- **Household Clusters** – groups households by usage behavior
- **Optimization Insights** – plain-English, data-driven energy-saving tips
- **Export Report** – download a summary report and cluster data

---

## 📁 About the Dataset

- **Source:** [Smart Meters in London (Kaggle)](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london)
- ~5,500 London households
- Half-hourly and daily energy readings, Nov 2011 – Feb 2014
- Enriched with ACORN demographic categories, UK bank holidays, and daily/hourly weather data

### 📥 Getting the Dataset

The raw dataset isn't included in this repo (it's a few GB, well past what's practical to version control). To get it:

1. Download it from Kaggle: 👉 [https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london)
2. Extract it into a `DATA_SET` folder in the project root.
3. Point the notebook's data-loading cell (`base_folder`) to that extracted folder.

> ⚠️ A free Kaggle account is required to download the dataset.

---

## 🛠️ Tech Stack

- Python, Pandas, NumPy, Scikit-learn (Random Forest)
- Streamlit + Plotly for the interactive dashboard
- Open-Meteo API for live weather forecasting

---
