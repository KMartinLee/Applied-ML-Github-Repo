import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.sparse import hstack
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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

# Initialize performance tracking
performance_summary = []

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
                      evals=[(dtest, 'eval')], early_stopping_rounds=10, verbose_eval=False)

    y_pred_probs = model.predict(dtest)
    y_pred_enc = y_pred_probs.argmax(axis=1)
    y_pred_labels = le.inverse_transform(y_pred_enc)
    y_test_labels = le.inverse_transform(y_test_enc)

    # Generate detailed metrics
    report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
    conf = confusion_matrix(y_test_labels, y_pred_labels)
    acc = accuracy_score(y_test_labels, y_pred_labels)

    # Collect performance metrics
    performance_summary.append({
        'Horizon': target_horizon,
        'Accuracy': acc,
        'Precision_Buy': report.get('Buy', {}).get('precision', np.nan),
        'Precision_Sell': report.get('Sell', {}).get('precision', np.nan),
        'Precision_Neutral': report.get('Neutral', {}).get('precision', np.nan),
        'F1_Buy': report.get('Buy', {}).get('f1-score', np.nan),
        'F1_Sell': report.get('Sell', {}).get('f1-score', np.nan),
        'F1_Neutral': report.get('Neutral', {}).get('f1-score', np.nan)
    })

    # Save detailed results
    with open("xgboost_results.txt", "a") as f:
        f.write(f"\n{'='*40}\nResults for: {target_horizon}\n{'='*40}\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test_labels, y_pred_labels) + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(conf, separator=', ') + "\n")
        f.write(f"Accuracy: {acc:.4f}\n")

# Convert to DataFrame
performance_summary_df = pd.DataFrame(performance_summary)
performance_summary_df.to_csv("XGBoost_performance_summary.csv", index=False)

print("\n✅ Performance summary saved as 'XGBoost_performance_summary.csv'")

# ================================================== Enhanced Plotting ================================================== #

print("\n" + "="*60)
print("XGBOOST MULTI-HORIZON VISUALIZATION")
print("="*60)

# Display the performance summary table
print("\nXGBoost Performance Summary Across All Horizons:")
print("-" * 50)
print(performance_summary_df.round(4))

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Original plots with enhancements
plt.figure(figsize=(12, 5))
plt.plot(performance_summary_df['Horizon'], performance_summary_df['Accuracy'], marker='o', linewidth=2, markersize=8, color='purple')
plt.title("XGBoost Accuracy Across Market Impact Horizons", fontweight='bold', fontsize=14)
plt.xticks(rotation=45)
plt.ylabel("Accuracy", fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# Add value labels on points
for i, (x, y) in enumerate(zip(performance_summary_df['Horizon'], performance_summary_df['Accuracy'])):
    plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(performance_summary_df['Horizon'], performance_summary_df['F1_Buy'], label='F1 Score (Buy)', marker='o', linewidth=2, markersize=8, color='green')
plt.plot(performance_summary_df['Horizon'], performance_summary_df['F1_Sell'], label='F1 Score (Sell)', marker='s', linewidth=2, markersize=8, color='red')
plt.title("XGBoost F1 Scores for 'Buy' and 'Sell'", fontweight='bold', fontsize=14)
plt.xticks(rotation=45)
plt.ylabel("F1 Score", fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# ========== Enhanced Visualizations ==========

# 1. Performance Heatmap
plt.figure(figsize=(14, 8))
heatmap_data = performance_summary_df.set_index('Horizon')[['Accuracy', 'F1_Buy', 'F1_Sell', 'Precision_Buy', 'Precision_Sell']]

sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='RdYlGn', 
            cbar_kws={'label': 'Performance Score'}, linewidths=0.5)
plt.title('XGBoost Performance Heatmap Across All Horizons', fontweight='bold', fontsize=16)
plt.xlabel('Market Impact Horizon', fontweight='bold')
plt.ylabel('Performance Metric', fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Comprehensive Performance Dashboard
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('XGBoost Performance Dashboard', fontsize=16, fontweight='bold')

# Subplot 1: Accuracy with trend
axes[0, 0].plot(performance_summary_df['Horizon'], performance_summary_df['Accuracy'], 
                marker='o', linewidth=3, markersize=10, color='purple')
axes[0, 0].set_title("Accuracy Across Horizons", fontweight='bold')
axes[0, 0].set_ylabel("Accuracy", fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim(0, 1)

# Subplot 2: F1 Scores
axes[0, 1].plot(performance_summary_df['Horizon'], performance_summary_df['F1_Buy'], 
                label='F1 Buy', marker='o', linewidth=2, markersize=8, color='green')
axes[0, 1].plot(performance_summary_df['Horizon'], performance_summary_df['F1_Sell'], 
                label='F1 Sell', marker='s', linewidth=2, markersize=8, color='red')
axes[0, 1].set_title("F1 Scores by Class", fontweight='bold')
axes[0, 1].set_ylabel("F1 Score", fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(0, 1)

# Subplot 3: Precision Comparison
axes[1, 0].plot(performance_summary_df['Horizon'], performance_summary_df['Precision_Buy'], 
                label='Precision Buy', marker='o', linewidth=2, markersize=8, color='darkgreen')
axes[1, 0].plot(performance_summary_df['Horizon'], performance_summary_df['Precision_Sell'], 
                label='Precision Sell', marker='s', linewidth=2, markersize=8, color='darkred')
axes[1, 0].set_title("Precision by Class", fontweight='bold')
axes[1, 0].set_ylabel("Precision", fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim(0, 1)

# Subplot 4: Best vs Worst Performing Horizons
if len(performance_summary_df) > 0:
    best_accuracy_idx = performance_summary_df['Accuracy'].idxmax()
    worst_accuracy_idx = performance_summary_df['Accuracy'].idxmin()

    performance_metrics = ['Accuracy', 'F1_Buy', 'F1_Sell', 'Precision_Buy', 'Precision_Sell']
    best_values = [performance_summary_df.loc[best_accuracy_idx, metric] for metric in performance_metrics]
    worst_values = [performance_summary_df.loc[worst_accuracy_idx, metric] for metric in performance_metrics]

    x_pos = range(len(performance_metrics))
    width = 0.35

    axes[1, 1].bar([x - width/2 for x in x_pos], best_values, width, 
                   label=f'Best ({performance_summary_df.loc[best_accuracy_idx, "Horizon"]})', color='lightgreen', alpha=0.8)
    axes[1, 1].bar([x + width/2 for x in x_pos], worst_values, width, 
                   label=f'Worst ({performance_summary_df.loc[worst_accuracy_idx, "Horizon"]})', color='lightcoral', alpha=0.8)

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
horizons_clean = [h.replace('MI_', '').replace('_MidClose', '') for h in performance_summary_df['Horizon']]

plt.plot(horizons_clean, performance_summary_df['Accuracy'], 
         marker='o', linewidth=3, markersize=10, label='Accuracy', color='purple')
plt.plot(horizons_clean, performance_summary_df['F1_Buy'], 
         marker='s', linewidth=2, markersize=8, label='F1 Buy', color='green')
plt.plot(horizons_clean, performance_summary_df['F1_Sell'], 
         marker='^', linewidth=2, markersize=8, label='F1 Sell', color='red')

plt.title('XGBoost Performance Trends Across Time Horizons', fontweight='bold', fontsize=14)
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
print("XGBOOST PERFORMANCE SUMMARY STATISTICS")
print("="*60)

if len(performance_summary_df) > 0:
    print(f"Best Overall Accuracy: {performance_summary_df['Accuracy'].max():.4f} at {performance_summary_df.loc[performance_summary_df['Accuracy'].idxmax(), 'Horizon']}")
    print(f"Worst Overall Accuracy: {performance_summary_df['Accuracy'].min():.4f} at {performance_summary_df.loc[performance_summary_df['Accuracy'].idxmin(), 'Horizon']}")
    print(f"Mean Accuracy Across All Horizons: {performance_summary_df['Accuracy'].mean():.4f}")
    print(f"Standard Deviation: {performance_summary_df['Accuracy'].std():.4f}")

    print(f"\nBest F1 Buy Score: {performance_summary_df['F1_Buy'].max():.4f} at {performance_summary_df.loc[performance_summary_df['F1_Buy'].idxmax(), 'Horizon']}")
    print(f"Best F1 Sell Score: {performance_summary_df['F1_Sell'].max():.4f} at {performance_summary_df.loc[performance_summary_df['F1_Sell'].idxmax(), 'Horizon']}")

    # Identify best performing horizon overall
    performance_summary_df['Overall_Score'] = (performance_summary_df['Accuracy'] + 
                                              performance_summary_df['F1_Buy'] + 
                                              performance_summary_df['F1_Sell']) / 3

    best_overall_idx = performance_summary_df['Overall_Score'].idxmax()
    print(f"\nBest Overall Performing Horizon: {performance_summary_df.loc[best_overall_idx, 'Horizon']}")
    print(f"Overall Score: {performance_summary_df.loc[best_overall_idx, 'Overall_Score']:.4f}")

    # Performance consistency analysis
    print(f"\nPerformance Consistency:")
    print(f"Accuracy Range: {performance_summary_df['Accuracy'].max() - performance_summary_df['Accuracy'].min():.4f}")
    print(f"F1 Buy Range: {performance_summary_df['F1_Buy'].max() - performance_summary_df['F1_Buy'].min():.4f}")
    print(f"F1 Sell Range: {performance_summary_df['F1_Sell'].max() - performance_summary_df['F1_Sell'].min():.4f}")

    if performance_summary_df['Accuracy'].std() < 0.05:
        print("✅ Low variance - XGBoost performance is consistent across horizons")
    else:
        print("⚠️  High variance - XGBoost performance varies significantly across horizons")

print("\n" + "="*60)
print("XGBOOST VISUALIZATION COMPLETED!")
print("="*60)