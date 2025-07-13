import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# -------------------------------------------------- Text Cleaning -------------------------------------------------- #
def remove_emojis(text):
    if pd.isna(text): return text
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251"
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

# -------------------------------------------------- Load Dataset -------------------------------------------------- #
file_path = "tweet_market_impact.xlsx"
df = pd.read_excel(file_path)
df['Tweet'] = df['Tweet'].apply(clean_tweet_text)
df = df[df['Tweet'].str.len() > 0]

mi_cols = [col for col in df.columns if col.startswith("MI_") and 'MidClose' in col]

# -------------------------------------------------- Model Definition -------------------------------------------------- #
class LSTMMarketModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, account_count):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.account_emb = nn.Embedding(account_count, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, tweet_seq, account_ids):
        lstm_out, _ = self.lstm(tweet_seq)
        lstm_feat = lstm_out[:, -1, :]
        acc_feat = self.account_emb(account_ids)
        combined = torch.cat((lstm_feat, acc_feat), dim=1)
        return self.classifier(combined)

# -------------------------------------------------- Training Utility -------------------------------------------------- #
def train_model(model, X_train, acc_train, y_train, X_test, acc_test, y_test, epochs=10, lr=1e-3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train, acc_train)
        loss = loss_fn(outputs, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test, acc_test).argmax(dim=1)
    return preds.cpu().numpy()

# -------------------------------------------------- Main Pipeline Loop -------------------------------------------------- #
performance_summary = []

for target_horizon in mi_cols:
    print(f"\n{'='*30}\nProcessing horizon: {target_horizon}\n{'='*30}")
    df_temp = df.dropna(subset=[target_horizon, 'Tweet', 'Twitter_acc']).copy()

    df_temp['Label'] = df_temp[target_horizon].apply(
        lambda x: 'Buy' if x > 0.0005 else ('Sell' if x < -0.0005 else 'Neutral')
    )

    if df_temp['Label'].nunique() < 2 or len(df_temp) < 200:
        print("Skipped due to insufficient data or label diversity.")
        continue

    df_temp = df_temp.sort_values(by=['Twitter_acc', 'Timestamp'])
    label_enc = LabelEncoder()
    df_temp['Label_enc'] = label_enc.fit_transform(df_temp['Label'])
    df_temp['acc_id'] = LabelEncoder().fit_transform(df_temp['Twitter_acc'])

    tfidf = TfidfVectorizer(max_features=300, stop_words='english')
    X_text = tfidf.fit_transform(df_temp['Tweet'])
    acc_ids = df_temp['acc_id'].values
    labels = df_temp['Label_enc'].values

    SEQ_LEN = 5
    X_seq, y_seq, acc_seq = [], [], []

    for acc in np.unique(acc_ids):
        idx = np.where(acc_ids == acc)[0]
        if len(idx) < SEQ_LEN: continue
        for i in range(len(idx) - SEQ_LEN + 1):
            ix = idx[i:i+SEQ_LEN]
            X_seq.append(X_text[ix].toarray())
            y_seq.append(labels[ix[-1]])
            acc_seq.append(acc)

    X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_seq = torch.tensor(y_seq, dtype=torch.long)
    acc_seq = torch.tensor(np.array(acc_seq), dtype=torch.long)

    X_tr, X_te, y_tr, y_te, acc_tr, acc_te = train_test_split(
        X_seq, y_seq, acc_seq, test_size=0.2, stratify=y_seq, random_state=42
    )

    model = LSTMMarketModel(input_dim=X_seq.shape[2], hidden_dim=64, account_count=len(np.unique(acc_ids)))
    preds = train_model(model, X_tr, acc_tr, y_tr, X_te, acc_te, y_te)

    acc = accuracy_score(y_te, preds)
    rep = classification_report(y_te, preds, target_names=label_enc.classes_, output_dict=True)
    conf = confusion_matrix(y_te, preds)

    '''with open("lstm_results.txt", "a") as f:
        f.write(f"\n{'='*40}\nResults for: {target_horizon}\n{'='*40}\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_te, preds, target_names=label_enc.classes_))
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(conf, separator=', '))
        f.write(f"\nAccuracy: {acc:.4f}\n")'''


    performance_summary.append({
        'Horizon': target_horizon,
        'Accuracy': acc,
        'Precision_Buy': rep.get('Buy', {}).get('precision', 0),
        'Precision_Sell': rep.get('Sell', {}).get('precision', 0),
        'F1_Buy': rep.get('Buy', {}).get('f1-score', 0),
        'F1_Sell': rep.get('Sell', {}).get('f1-score', 0)
    })

# -------------------------------------------------- Plot Results -------------------------------------------------- #
summary_df = pd.DataFrame(performance_summary)

plt.figure(figsize=(10, 5))
plt.plot(summary_df['Horizon'], summary_df['Accuracy'], marker='o')
plt.title("LSTM Accuracy Across Market Impact Horizons")
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(summary_df['Horizon'], summary_df['F1_Buy'], label='F1 Score (Buy)', marker='o')
plt.plot(summary_df['Horizon'], summary_df['F1_Sell'], label='F1 Score (Sell)', marker='o')
plt.title("LSTM F1 Scores for 'Buy' and 'Sell'")
plt.xticks(rotation=45)
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
