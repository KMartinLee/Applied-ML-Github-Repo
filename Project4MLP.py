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

# ========== Step 4: MLP Performance Plotting with Heatmap ==========
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "="*60)
print("MLP MULTI-HORIZON VISUALIZATION")
print("="*60)

# Display the performance summary table
print("\nMLP Performance Summary Across All Horizons:")
print("-" * 50)
print(summary_df.round(4))

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Original plots with enhancements
plt.figure(figsize=(12, 5))
plt.plot(summary_df['Horizon'], summary_df['Accuracy'], marker='o', linewidth=2, markersize=8, color='navy')
plt.title("MLP Accuracy Across Market Impact Horizons", fontweight='bold', fontsize=14)
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
plt.title("MLP F1 Scores for 'Buy' and 'Sell'", fontweight='bold', fontsize=14)
plt.xticks(rotation=45)
plt.ylabel("F1 Score", fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# ========== Further Visualizations ==========

# 1. Performance Heatmap (like the Random Forest version)
plt.figure(figsize=(14, 8))
heatmap_data = summary_df.set_index('Horizon')[['Accuracy', 'F1_Buy', 'F1_Sell', 'Precision_Buy', 'Precision_Sell']]

sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='RdYlGn', 
            cbar_kws={'label': 'Performance Score'}, linewidths=0.5)
plt.title('MLP Performance Heatmap Across All Horizons', fontweight='bold', fontsize=16)
plt.xlabel('Market Impact Horizon', fontweight='bold')
plt.ylabel('Performance Metric', fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Comprehensive Performance Dashboard
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('MLP Neural Network Performance Dashboard', fontsize=16, fontweight='bold')

# Subplot 1: Accuracy with trend
axes[0, 0].plot(summary_df['Horizon'], summary_df['Accuracy'], 
                marker='o', linewidth=3, markersize=10, color='navy')
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
         marker='o', linewidth=3, markersize=10, label='Accuracy', color='navy')
plt.plot(horizons_clean, summary_df['F1_Buy'], 
         marker='s', linewidth=2, markersize=8, label='F1 Buy', color='green')
plt.plot(horizons_clean, summary_df['F1_Sell'], 
         marker='^', linewidth=2, markersize=8, label='F1 Sell', color='red')

plt.title('MLP Performance Trends Across Time Horizons', fontweight='bold', fontsize=14)
plt.xlabel('Time Horizon', fontweight='bold')
plt.ylabel('Performance Score', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# ========== Performance Analysis Summary ==========
print("\n" + "="*60)
print("MLP PERFORMANCE SUMMARY STATISTICS")
print("="*60)

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
    print("✅ Low variance - MLP performance is consistent across horizons")
else:
    print("⚠️  High variance - MLP performance varies significantly across horizons")

print("\n" + "="*60)
print("MLP VISUALIZATION COMPLETED!")
print("="*60)