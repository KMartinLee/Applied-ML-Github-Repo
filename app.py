import streamlit as st
import pandas as pd
from PIL import Image
import os


st.set_page_config(page_title="ML Dashboard", layout="wide")

st.title("Market sentiment and market impact prediction models on crude oil prices")
st.markdown("""
### 📊 Custom Tweet-Based Dataset for Crude Oil Market Analysis

We constructed a unique dataset to study the short-term impact of tweets on crude oil prices. Below is an overview:

---

- 🧾 **Data Source:**  
  Collected tweets from curated Twitter accounts influential in the energy and financial sectors.""")


accounts = [
    "BurggrabenH_tweets", "IDF_tweets", "IEA_tweets", "IRIran_Military_tweets",
    "IntelDoge_tweets", "IsraeliPM_tweets", "JavierBlas_tweets", "OPECSecretariat_tweets",
    "OSE_Yemen_tweets", "OilSheppard_tweets", "POTUS_tweets", "SStapczynski_tweets",
    "TrumpDailyPosts_tweets", "WhiteHouse_tweets", "Yemenimilitary_tweets",
    "chigrl_tweets", "mfa_russia_tweets", "realDonaldTrump_tweets",
    "sentdefender_tweets", "zerohedge_tweets"
]

df = pd.DataFrame(accounts, columns=["Twitter Accounts"])
st.table(df)

st.markdown("""
- 🕒 **Timeframe:**  
  Tweets are time-aligned with **market price data**—capturing price movement from **1 minute up to 2 days** after each tweet.""")

timeframe = pd.DataFrame({
    "Timeframe": [
        "1 minute", "5 minutes", "15 minutes", "30 minutes",
        "1 hour", "2 hours", "4 hours", "6 hours",
        "12 hours", "1 day", "2 days"
    ]
})
df2 = pd.DataFrame(timeframe)
st.table(df2)

st.markdown("""
- 🧠 **Labeling Mechanism:**  
  For each tweet, we computed the **market impact** based on price returns and assigned a **Buy / Neutral / Sell** label:
    - **Buy:** Return > +5 basis points  
    - **Sell:** Return < -5 basis points  
    - **Neutral:** Between -5 and +5 basis points

            
- 🔍 **Structure of the Dataset (`tweet_market_impact.xlsx`):**
  - `Tweet`: Raw tweet content  
  - `Timestamp`: Time of tweet  
  - `Twitter_acc`: Source Twitter account  
  - `MI_*`: Market impact values over various horizons  
    *(e.g., `MI_1min_MidClose`, `MI_5min_MidClose`, ..., `MI_2d_MidClose`)*

  """)

excel_screen_path = os.path.join("images", "Screenshot 2025-07-13 at 23.22.06.png")
excel_screen = Image.open(excel_screen_path)
st.image(excel_screen, caption="Excel Screenshot", use_container_width=True)

st.markdown("""
- 🧹 **Data Cleaning Includes:**
  - Removing emojis and hyperlinks  
  - Filtering out empty or non-informative tweets
  - Removed hyperlinks
  - Empty Tweet Removal          

---

Now, by using the sidebar, we are going to explore the different models and their results:
- XGBoost 📈  
- LSTM 🧠  
- MLP 🔬  
- Random Forest 🌳
""")

