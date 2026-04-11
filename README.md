# Demand Forecast Studio

A Streamlit-based demand forecasting tool for FMCG teams. Upload your master data, filter by any dimension combination, and compare 5 statistical forecasting methods.

## Features
- **Upload** .xlsx master data with monthly time series
- **Filter** by UOM, Channel, Region, ZSM, Category, Brand, Sub Brand, Track, AOP-Track
- **5 forecasting methods**: Holt-Winters, SARIMA, SES, Linear Regression + Seasonal Dummies, Weighted Moving Average
- **Auto-recommends** the best method based on lowest MAPE
- **Interactive Plotly charts** with actual vs fitted vs forecast
- **Export results** to Excel (historical + fitted, forecast, accuracy metrics)

## Deployment on Streamlit Community Cloud (Free)

### Step 1: Create a GitHub repo
1. Go to [github.com](https://github.com) and create a new repository (e.g. `demand-forecast-studio`)
2. Upload these 3 files to the repo:
   - `app.py`
   - `requirements.txt`
   - `README.md`

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **New app**
4. Select your repo, branch (`main`), and file (`app.py`)
5. Click **Deploy**
6. Your app will be live at `https://your-app-name.streamlit.app`

### Step 3: Share with your team
- Share the URL with your 6 team members
- They just open the link, upload the data file, and start forecasting
- No installation required

## Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```
