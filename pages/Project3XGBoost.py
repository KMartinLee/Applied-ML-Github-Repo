import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.sparse import hstack
import xgboost as xgb
#import shap

# -------------------------------------------------- Text Cleaning -------------------------------------------------- #
def remove_emojis(text):
    if pd.isna(text): return text
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_hyperlinks(text):
    if pd.isna(text): return text
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'\b(?:t\.co|bit\.ly|tinyurl\.com|goo\.gl|ow\.ly|short\.link)/\S+', '', text)
    return text

def clean_tweet_text(text):
    if pd.isna(text): return text
    text = remove_hyperlinks(text)
    text = remove_emojis(text)
    return re.sub(r'\s+', ' ', text).strip()

# -------------------------------------------------- Load and Preprocess -------------------------------------------------- #
file_path = "tweet_market_impact.xlsx"
df = pd.read_excel(file_path, sheet_name=0)

df['Tweet_Original'] = df['Tweet']
df['Tweet'] = df['Tweet'].apply(clean_tweet_text)
df = df[df['Tweet'].str.len() > 0]

# Identify all MI columns
mi_cols = [col for col in df.columns if col.startswith("MI_") and 'MidClose' in col]

# -------------------------------------------------- Main Pipeline Loop -------------------------------------------------- #
for target_horizon in mi_cols:
    print(f"\n{'='*30}\nProcessing horizon: {target_horizon}\n{'='*30}")

    df_clean = df.dropna(subset=[target_horizon, 'Tweet', 'Twitter_acc']).copy()

    df_clean['Label'] = df_clean[target_horizon].apply(
        lambda x: 'Buy' if x > 0.0005 else ('Sell' if x < -0.0005 else 'Neutral')
    )

    # Skip if one class dominates or too few samples
    if df_clean['Label'].nunique() < 2 or df_clean.shape[0] < 100:
        print("Skipped due to insufficient data or label diversity.")
        continue

    # TF-IDF + OHE
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    X_text = tfidf.fit_transform(df_clean['Tweet'])

    ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    X_account = ohe.fit_transform(df_clean[['Twitter_acc']])

    X = hstack([X_text, X_account])
    y = df_clean['Label']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    # Train XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train_enc)
    dtest = xgb.DMatrix(X_test, label=y_test_enc)

    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'verbosity': 0
    }

    model = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=100,
                      evals=[(dtest, 'eval')], early_stopping_rounds=10)

    y_pred_probs = model.predict(dtest)
    y_pred_enc = y_pred_probs.argmax(axis=1)
    y_pred_labels = le.inverse_transform(y_pred_enc)

    report = classification_report(le.inverse_transform(y_test_enc), y_pred_labels)
    conf = confusion_matrix(le.inverse_transform(y_test_enc), y_pred_labels)
    acc = accuracy_score(le.inverse_transform(y_test_enc), y_pred_labels)

    with open("xgboost_results.txt", "a") as f:
        f.write(f"\n{'='*40}\nResults for: {target_horizon}\n{'='*40}\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(conf, separator=', ') + "\n")
        f.write(f"Accuracy: {acc:.4f}\n")


    # -------------------------------------------------- SHAP Feature Importances -------------------------------------------------- #
    '''explainer = shap.TreeExplainer(model)
    X_test_dense = X_test.toarray()
    shap_values = explainer.shap_values(X_test_dense)

    tfidf_features = list(tfidf.get_feature_names_out())
    account_features = list(ohe.get_feature_names_out(['Twitter_acc']))
    feature_names = tfidf_features + account_features

    if isinstance(shap_values, list):
        shap_array = np.array([np.abs(sv).mean(axis=0) for sv in shap_values])
        mean_abs_shap = shap_array.mean(axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    top_features_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values(by='mean_abs_shap', ascending=False).head(10)

    print("Top 10 SHAP Features:")
    print(top_features_df.to_string(index=False))
'''

import matplotlib.pyplot as plt

# Placeholder: initialize storage for summary results
performance_summary = []

# We'll simulate what would be collected from the loop:
# Let's create dummy values for illustration purposes.
# In practice, these would be collected inside the loop in the actual code.

performance_summary = pd.DataFrame({
    'Horizon': [
        'MI_1min_MidClose', 'MI_5min_MidClose', 'MI_15min_MidClose',
        'MI_30min_MidClose', 'MI_1h_MidClose', 'MI_2h_MidClose',
        'MI_4h_MidClose', 'MI_8h_MidClose', 'MI_12h_MidClose',
        'MI_1d_MidClose', 'MI_2d_MidClose'
    ],
    'Accuracy': [0.55, 0.58, 0.61, 0.60, 0.62, 0.59, 0.60, 0.61, 0.63, 0.60, 0.58],
    'Precision_Buy': [0.56, 0.60, 0.63, 0.62, 0.64, 0.60, 0.61, 0.62, 0.65, 0.61, 0.59],
    'Precision_Sell': [0.52, 0.55, 0.58, 0.57, 0.59, 0.56, 0.57, 0.58, 0.60, 0.57, 0.54],
    'F1_Buy': [0.54, 0.57, 0.60, 0.59, 0.61, 0.58, 0.59, 0.60, 0.62, 0.59, 0.56],
    'F1_Sell': [0.50, 0.53, 0.56, 0.55, 0.57, 0.54, 0.55, 0.56, 0.58, 0.55, 0.52]
})

# Plotting Accuracy across Horizons
'''plt.figure(figsize=(10, 5))
plt.plot(performance_summary['Horizon'], performance_summary['Accuracy'], marker='o')
plt.title("Model Accuracy Across Market Impact Horizons")
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting F1 Scores for Buy and Sell
plt.figure(figsize=(10, 5))
plt.plot(performance_summary['Horizon'], performance_summary['F1_Buy'], label='F1 Score (Buy)', marker='o')
plt.plot(performance_summary['Horizon'], performance_summary['F1_Sell'], label='F1 Score (Sell)', marker='o')
plt.title("F1 Scores for 'Buy' and 'Sell' Classes")
plt.xticks(rotation=45)
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()'''



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 XGBoost Model Performance")

@st.cache_data
def load_data():
    return pd.read_csv("xgboost_performance_summary.csv")

df = load_data()

st.dataframe(df, use_container_width=True)

# Accuracy
st.subheader("Model Accuracy Across Horizons")
fig1, ax1 = plt.subplots()
ax1.plot(df['Horizon'], df['Accuracy'], marker='o')
ax1.set_title("Accuracy")
ax1.set_ylabel("Accuracy")
ax1.set_xticks(range(len(df)))
ax1.set_xticklabels(df['Horizon'], rotation=45)
ax1.grid(True)
st.pyplot(fig1)

# F1 Scores
st.subheader("F1 Scores (Buy vs Sell)")
fig2, ax2 = plt.subplots()
ax2.plot(df['Horizon'], df['F1_Buy'], label='F1 Buy', marker='o')
ax2.plot(df['Horizon'], df['F1_Sell'], label='F1 Sell', marker='o')
ax2.set_title("F1 Scores")
ax2.set_ylabel("F1 Score")
ax2.set_xticks(range(len(df)))
ax2.set_xticklabels(df['Horizon'], rotation=45)
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)
