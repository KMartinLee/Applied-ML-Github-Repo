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
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )

    def forward(self, tweet_seq, account_ids):
        lstm_out, _ = self.lstm(tweet_seq)
        lstm_feat = lstm_out[:, -1, :]
        acc_feat = self.account_emb(account_ids)
        combined = torch.cat((lstm_feat, acc_feat), dim=1)
        return self.classifier(combined)

# -------------------------------------------------- Training Utility -------------------------------------------------- #
def train_model(model, X_train, acc_train, y_train, X_test, acc_test, y_test, epochs=20, lr=1e-3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    print(f"Training LSTM for {epochs} epochs...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train, acc_train)
        loss = loss_fn(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

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

    # Sort by account and timestamp for sequence creation
    if 'Timestamp' in df_temp.columns:
        df_temp = df_temp.sort_values(by=['Twitter_acc', 'Timestamp'])
    else:
        df_temp = df_temp.sort_values(by=['Twitter_acc'])
    
    label_enc = LabelEncoder()
    df_temp['Label_enc'] = label_enc.fit_transform(df_temp['Label'])
    
    acc_enc = LabelEncoder()
    df_temp['acc_id'] = acc_enc.fit_transform(df_temp['Twitter_acc'])

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

    if len(X_seq) < 50:  # Need minimum sequences for training
        print("Insufficient sequences for training.")
        continue

    X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_seq = torch.tensor(y_seq, dtype=torch.long)
    acc_seq = torch.tensor(np.array(acc_seq), dtype=torch.long)

    X_tr, X_te, y_tr, y_te, acc_tr, acc_te = train_test_split(
        X_seq, y_seq, acc_seq, test_size=0.2, stratify=y_seq, random_state=42
    )

    model = LSTMMarketModel(input_dim=X_seq.shape[2], hidden_dim=64, 
                           account_count=len(np.unique(acc_ids)))
    preds = train_model(model, X_tr, acc_tr, y_tr, X_te, acc_te, y_te)

    acc = accuracy_score(y_te, preds)
    rep = classification_report(y_te, preds, target_names=label_enc.classes_, output_dict=True)
    conf = confusion_matrix(y_te, preds)

    # Save detailed results
    with open("lstm_results.txt", "a") as f:
        f.write(f"\n{'='*40}\nResults for: {target_horizon}\n{'='*40}\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_te, preds, target_names=label_enc.classes_))
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(conf, separator=', '))
        f.write(f"\nAccuracy: {acc:.4f}\n")

    # Collect performance metrics
    performance_summary.append({
        'Horizon': target_horizon,
        'Accuracy': acc,
        'Precision_Buy': rep.get('Buy', {}).get('precision', np.nan),
        'Precision_Sell': rep.get('Sell', {}).get('precision', np.nan),
        'Precision_Neutral': rep.get('Neutral', {}).get('precision', np.nan),
        'F1_Buy': rep.get('Buy', {}).get('f1-score', np.nan),
        'F1_Sell': rep.get('Sell', {}).get('f1-score', np.nan),
        'F1_Neutral': rep.get('Neutral', {}).get('f1-score', np.nan)
    })

# Convert to DataFrame
summary_df = pd.DataFrame(performance_summary)
summary_df.to_csv("LSTM_performance_summary.csv", index=False)

print("\n✅ Performance summary saved as 'LSTM_performance_summary.csv'")

# ================================================== Enhanced Plotting ================================================== #

print("\n" + "="*60)
print("LSTM MULTI-HORIZON VISUALIZATION")
print("="*60)

# Display the performance summary table
print("\nLSTM Performance Summary Across All Horizons:")
print("-" * 50)
print(summary_df.round(4))

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Original plots with enhancements
plt.figure(figsize=(12, 5))
plt.plot(summary_df['Horizon'], summary_df['Accuracy'], marker='o', linewidth=2, markersize=8, color='darkblue')
plt.title("LSTM Accuracy Across Market Impact Horizons", fontweight='bold', fontsize=14)
plt.xticks(rotation=45)
plt.ylabel("Accuracy", fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# Add value labels on points
for i, (x, y) in enumerate(zip(summary_df['Horizon'], summary_df['Accuracy'])):
    plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(summary_df['Horizon'], summary_df['F1_Buy'], label='F1 Score (Buy)', marker='o', linewidth=2, markersize=8, color='green')
plt.plot(summary_df['Horizon'], summary_df['F1_Sell'], label='F1 Score (Sell)', marker='s', linewidth=2, markersize=8, color='red')
plt.title("LSTM F1 Scores for 'Buy' and 'Sell'", fontweight='bold', fontsize=14)
plt.xticks(rotation=45)
plt.ylabel("F1 Score", fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# ========== NEW: Enhanced Visualizations ==========

# 1. Performance Heatmap
plt.figure(figsize=(14, 8))
heatmap_data = summary_df.set_index('Horizon')[['Accuracy', 'F1_Buy', 'F1_Sell', 'Precision_Buy', 'Precision_Sell']]

sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='RdYlGn', 
            cbar_kws={'label': 'Performance Score'}, linewidths=0.5)
plt.title('LSTM Performance Heatmap Across All Horizons', fontweight='bold', fontsize=16)
plt.xlabel('Market Impact Horizon', fontweight='bold')
plt.ylabel('Performance Metric', fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Comprehensive Performance Dashboard
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('LSTM Deep Learning Performance Dashboard', fontsize=16, fontweight='bold')

# Subplot 1: Accuracy with trend
axes[0, 0].plot(summary_df['Horizon'], summary_df['Accuracy'], 
                marker='o', linewidth=3, markersize=10, color='darkblue')
axes[0, 0].set_title("Accuracy Across Horizons", fontweight='bold')
axes[0, 0].set_ylabel("Accuracy", fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim(0, 1)

# Subplot 2: F1 Scores
axes[0, 1].plot(summary_df['Horizon'], summary_df['F1_Buy'], 
                label='F1 Buy', marker='o', linewidth=2, markersize=8, color='green')
axes[0, 1].plot(summary_df['Horizon'], summary_df['F1_Sell'], 
                label='F1 Sell', marker='s', linewidth=2, markersize=8, color='red')
axes[0, 1].set_title("F1 Scores by Class", fontweight='bold')
axes[0, 1].set_ylabel("F1 Score", fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(0, 1)

# Subplot 3: Precision Comparison
axes[1, 0].plot(summary_df['Horizon'], summary_df['Precision_Buy'], 
                label='Precision Buy', marker='o', linewidth=2, markersize=8, color='darkgreen')
axes[1, 0].plot(summary_df['Horizon'], summary_df['Precision_Sell'], 
                label='Precision Sell', marker='s', linewidth=2, markersize=8, color='darkred')
axes[1, 0].set_title("Precision by Class", fontweight='bold')
axes[1, 0].set_ylabel("Precision", fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim(0, 1)

# Subplot 4: Best vs Worst Performing Horizons
if len(summary_df) > 0:
    best_accuracy_idx = summary_df['Accuracy'].idxmax()
    worst_accuracy_idx = summary_df['Accuracy'].idxmin()

    performance_metrics = ['Accuracy', 'F1_Buy', 'F1_Sell', 'Precision_Buy', 'Precision_Sell']
    best_values = [summary_df.loc[best_accuracy_idx, metric] for metric in performance_metrics]
    worst_values = [summary_df.loc[worst_accuracy_idx, metric] for metric in performance_metrics]

    x_pos = range(len(performance_metrics))
    width = 0.35

    axes[1, 1].bar([x - width/2 for x in x_pos], best_values, width, 
                   label=f'Best ({summary_df.loc[best_accuracy_idx, "Horizon"]})', color='lightgreen', alpha=0.8)
    axes[1, 1].bar([x + width/2 for x in x_pos], worst_values, width, 
                   label=f'Worst ({summary_df.loc[worst_accuracy_idx, "Horizon"]})', color='lightcoral', alpha=0.8)

    axes[1, 1].set_title('Best vs Worst Performing Horizons', fontweight='bold')
    axes[1, 1].set_ylabel('Score', fontweight='bold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(performance_metrics, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 1)

    # Add value labels on bars
    for i, (best, worst) in enumerate(zip(best_values, worst_values)):
        axes[1, 1].text(i - width/2, best + 0.02, f'{best:.3f}', ha='center', va='bottom', fontsize=8)
        axes[1, 1].text(i + width/2, worst + 0.02, f'{worst:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# 3. Time Horizon Trend Analysis
plt.figure(figsize=(12, 6))
horizons_clean = [h.replace('MI_', '').replace('_MidClose', '') for h in summary_df['Horizon']]

plt.plot(horizons_clean, summary_df['Accuracy'], 
         marker='o', linewidth=3, markersize=10, label='Accuracy', color='darkblue')
plt.plot(horizons_clean, summary_df['F1_Buy'], 
         marker='s', linewidth=2, markersize=8, label='F1 Buy', color='green')
plt.plot(horizons_clean, summary_df['F1_Sell'], 
         marker='^', linewidth=2, markersize=8, label='F1 Sell', color='red')

plt.title('LSTM Performance Trends Across Time Horizons', fontweight='bold', fontsize=14)
plt.xlabel('Time Horizon', fontweight='bold')
plt.ylabel('Performance Score', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# 4. LSTM-Specific Analysis: Model Complexity vs Performance
plt.figure(figsize=(10, 6))
sequence_lengths = [5] * len(summary_df)  # Your sequence length
hidden_dims = [64] * len(summary_df)      # Your hidden dimension

# Create a complexity scatter plot
plt.scatter(summary_df['Accuracy'], summary_df['F1_Buy'], 
           s=100, alpha=0.7, c='darkblue', label='F1 Buy vs Accuracy')
plt.scatter(summary_df['Accuracy'], summary_df['F1_Sell'], 
           s=100, alpha=0.7, c='red', label='F1 Sell vs Accuracy')

plt.xlabel('Accuracy', fontweight='bold')
plt.ylabel('F1 Score', fontweight='bold')
plt.title('LSTM: F1 Score vs Accuracy Correlation', fontweight='bold', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ========== Performance Analysis Summary ==========
print("\n" + "="*60)
print("LSTM PERFORMANCE SUMMARY STATISTICS")
print("="*60)

if len(summary_df) > 0:
    print(f"Best Overall Accuracy: {summary_df['Accuracy'].max():.4f} at {summary_df.loc[summary_df['Accuracy'].idxmax(), 'Horizon']}")
    print(f"Worst Overall Accuracy: {summary_df['Accuracy'].min():.4f} at {summary_df.loc[summary_df['Accuracy'].idxmin(), 'Horizon']}")
    print(f"Mean Accuracy Across All Horizons: {summary_df['Accuracy'].mean():.4f}")
    print(f"Standard Deviation: {summary_df['Accuracy'].std():.4f}")

    print(f"\nBest F1 Buy Score: {summary_df['F1_Buy'].max():.4f} at {summary_df.loc[summary_df['F1_Buy'].idxmax(), 'Horizon']}")
    print(f"Best F1 Sell Score: {summary_df['F1_Sell'].max():.4f} at {summary_df.loc[summary_df['F1_Sell'].idxmax(), 'Horizon']}")

    # Identify best performing horizon overall
    summary_df['Overall_Score'] = (summary_df['Accuracy'] + 
                                  summary_df['F1_Buy'] + 
                                  summary_df['F1_Sell']) / 3

    best_overall_idx = summary_df['Overall_Score'].idxmax()
    print(f"\nBest Overall Performing Horizon: {summary_df.loc[best_overall_idx, 'Horizon']}")
    print(f"Overall Score: {summary_df.loc[best_overall_idx, 'Overall_Score']:.4f}")

    # Performance consistency analysis
    print(f"\nPerformance Consistency:")
    print(f"Accuracy Range: {summary_df['Accuracy'].max() - summary_df['Accuracy'].min():.4f}")
    print(f"F1 Buy Range: {summary_df['F1_Buy'].max() - summary_df['F1_Buy'].min():.4f}")
    print(f"F1 Sell Range: {summary_df['F1_Sell'].max() - summary_df['F1_Sell'].min():.4f}")

    if summary_df['Accuracy'].std() < 0.05:
        print("✅ Low variance - LSTM performance is consistent across horizons")
    else:
        print("⚠️  High variance - LSTM performance varies significantly across horizons")

    # LSTM-specific insights
    print(f"\nLSTM Deep Learning Insights:")
    print(f"🧠 Model Architecture: LSTM + Account Embeddings")
    print(f"📊 Sequence Length: 5 tweets per prediction")
    print(f"🔗 Hidden Dimension: 64 neurons")
    print(f"⚡ Training: Adam optimizer with dropout regularization")

print("\n" + "="*60)
print("LSTM VISUALIZATION COMPLETED!")
print("="*60)