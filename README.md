# ⚡ Autonomous Energy Optimization Platform for Smart Grid

An AI-powered dashboard that forecasts household energy demand, segments customers by usage pattern, and generates optimization insights — built on real London smart meter data.

**🔗 Live Demo:** [Add your deployment link here]

---

## 🚀 How to Run

Follow these steps in order:

1. **Clone or download this project** to your local machine.

2. **Install the required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook first**
   Open `Autonomous_Energy_Optimization_Platform.ipynb` and execute all cells from top to bottom.
   This step trains the forecasting model and automatically creates an `outputs/` folder containing the following three files:
   - `energy_forecast_model.pkl`
   - `cleaned_merged_energy_data.csv`
   - `household_usage_groups.csv`

   > ⚠️ The dashboard will not run without these files, so this step must be completed first.

4. **Run the dashboard**
   Once the `outputs/` folder has been generated, launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   Ensure the `outputs/` folder is placed in the same directory as `app.py`.

```
Autonomous Energy Optimization Platform for Smart Grid/
├── main.ipynb
├── app.py
├── requirements.txt
├── README.md
├── DATA_SET.zip
└── outputs/
    ├── energy_forecast_model.pkl
    ├── cleaned_merged_energy_data.csv
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

The raw dataset (`DATA_SET.zip`) is included in this project. To use it:

1. Extract `DATA_SET.zip` into the project's root folder.
2. Point the notebook's data-loading cells to the extracted folder path.

Alternatively, download the dataset directly from Kaggle:
👉 [https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london)

> ⚠️ A free Kaggle account is required to download the dataset. Once downloaded, extract it into the project folder before running the notebook.

---

## 🛠️ Tech Stack

- Python, Pandas, NumPy, Scikit-learn (Random Forest)
- Streamlit + Plotly for the interactive dashboard
- Open-Meteo API for live weather forecasting
- 
---

## 📧 Contact:** [bhavintak8863@gmail.com]