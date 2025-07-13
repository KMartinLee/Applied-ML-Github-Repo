import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

# ========== Step 0: File Check ==========
file_path = "tweet_market_impact.xlsx"
if not os.path.exists(file_path):
    raise FileNotFoundError("Excel file not found. Ensure tweet_market_impact.xlsx is in your working directory.")

df = pd.read_excel(file_path)

# ========== Step 1: Cleaning ==========
def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_hyperlinks(text):
    text = re.sub(r"http[s]?://\S+", '', text)
    text = re.sub(r"www\.\S+", '', text)
    return re.sub(r'\b(?:t\.co|bit\.ly|tinyurl\.com|goo\.gl|ow\.ly|short\.link)/\S+', '', text)

def clean_tweet_text(text):
    if pd.isna(text): return text
    text = remove_hyperlinks(text)
    text = remove_emojis(text)
    return re.sub(r'\s+', ' ', text).strip()

df['Tweet'] = df['Tweet'].apply(clean_tweet_text)
df = df[df['Tweet'].str.len() > 0]

# ========== Step 2: Loop Through Horizons ==========
mi_cols = [col for col in df.columns if col.startswith("MI_") and "MidClose" in col]
performance_summary = []

label_map = {'Sell': 0, 'Neutral': 1, 'Buy': 2}
reverse_label_map = {v: k for k, v in label_map.items()}

for target_col in mi_cols:
    print(f"\n{'='*30}\nProcessing: {target_col}\n{'='*30}")

    df_temp = df.dropna(subset=[target_col, 'Tweet', 'Twitter_acc']).copy()

    # Labeling
    def classify(x):
        if x > 0.0005: return 'Buy'
        elif x < -0.0005: return 'Sell'
        else: return 'Neutral'

    df_temp['Label'] = df_temp[target_col].apply(classify)
    if df_temp['Label'].nunique() < 2 or len(df_temp) < 200:
        print("❌ Skipped (Insufficient label variety or sample size)")
        continue

    # Feature Engineering
    tfidf = TfidfVectorizer(max_features=2000, stop_words='english', min_df=2, max_df=0.95, ngram_range=(1,2))
    X_text = tfidf.fit_transform(df_temp['Tweet'])

    ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    X_acc = ohe.fit_transform(df_temp[['Twitter_acc']])

    X = hstack([X_text, X_acc])
    y = df_temp['Label'].map(label_map)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # MLP Model
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(150, 100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=False
    )

    mlp_model.fit(X_train, y_train)
    y_pred = mlp_model.predict(X_test)
    y_pred_proba = mlp_model.predict_proba(X_test)

    report = classification_report(y_test, y_pred, target_names=reverse_label_map.values(), output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)

    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    except:
        roc_auc = np.nan

    performance_summary.append({
        'Horizon': target_col,
        'Accuracy': accuracy,
        'Precision_Buy': report.get('Buy', {}).get('precision', np.nan),
        'Precision_Sell': report.get('Sell', {}).get('precision', np.nan),
        'F1_Buy': report.get('Buy', {}).get('f1-score', np.nan),
        'F1_Sell': report.get('Sell', {}).get('f1-score', np.nan)
    })

# ========== Step 3: Create DataFrame ==========
summary_df = pd.DataFrame(performance_summary)
summary_df.to_csv("mlp_performance_summary.csv", index=False)
print("\n✅ Performance summary saved as 'mlp_performance_summary.csv'")

# ========== Step 4: Plotting ==========
plt.figure(figsize=(10, 5))
plt.plot(summary_df['Horizon'], summary_df['Accuracy'], marker='o')
plt.title("MLP Accuracy Across Market Impact Horizons")
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(summary_df['Horizon'], summary_df['F1_Buy'], label='F1 Score (Buy)', marker='o')
plt.plot(summary_df['Horizon'], summary_df['F1_Sell'], label='F1 Score (Sell)', marker='o')
plt.title("MLP F1 Scores for 'Buy' and 'Sell'")
plt.xticks(rotation=45)
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
