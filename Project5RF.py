# import pandas as pd
# import re
# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.neural_network import MLPClassifier
# from scipy.sparse import hstack
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
# import numpy as np
# import warnings
# warnings.filterwarnings('ignore')

# #--------------------------------------------------step 0: Import File.--------------------------------------------------

# # Load the uploaded Excel file - Updated path for your system
# #file_path = "/Users/t.b.k.bihari/Documents/GitHub/Applied-ML-Github-Repo/tweet_market_impact.xlsx"
# file_path = "tweet_market_impact.xlsx"

# # Check if file exists before loading
# import os
# if not os.path.exists(file_path):
#     print(f"❌ File not found at: {file_path}")
#     print("Please check the file location and update the path.")
#     # Alternative: Look for the file in current directory
#     current_dir_file = "tweet_market_impact.xlsx"
#     if os.path.exists(current_dir_file):
#         print(f"✅ Found file in current directory: {current_dir_file}")
#         file_path = current_dir_file
#     else:
#         print("Please ensure the Excel file is in the same folder as your Python script.")
#         exit()

# xls = pd.ExcelFile(file_path)
# df = xls.parse('Sheet1')

# # Display the first few rows to understand the structure
# #print(df.head())

# #--------------------------------------------------step 1: Text Cleaning Functions.--------------------------------------------------
# def remove_emojis(text):
#     """Remove emojis from text using regex pattern"""
#     if pd.isna(text):
#         return text
#     # Pattern to match emoji characters
#     emoji_pattern = re.compile("["
#                               u"\U0001F600-\U0001F64F"  # emoticons
#                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#                               u"\U0001F680-\U0001F6FF"  # transport & map
#                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                               u"\U00002702-\U000027B0"  # dingbats
#                               u"\U000024C2-\U0001F251"  # enclosed characters
#                               "]+", flags=re.UNICODE)
#     return emoji_pattern.sub(r'', text)

# def remove_hyperlinks(text):
#     """Remove hyperlinks from text"""
#     if pd.isna(text):
#         return text
#     # Remove URLs starting with http, https, or www
#     url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
#     text = url_pattern.sub('', text)
    
#     # Remove www links
#     www_pattern = re.compile(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
#     text = www_pattern.sub('', text)
    
#     # Remove shortened URLs (like t.co, bit.ly, etc.)
#     short_url_pattern = re.compile(r'\b(?:t\.co|bit\.ly|tinyurl\.com|goo\.gl|ow\.ly|short\.link)/\S+')
#     text = short_url_pattern.sub('', text)
    
#     return text

# def clean_tweet_text(text):
#     """Clean tweet text by removing emojis and hyperlinks"""
#     if pd.isna(text):
#         return text
    
#     # Remove hyperlinks first
#     text = remove_hyperlinks(text)
    
#     # Remove emojis
#     text = remove_emojis(text)
    
#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     return text

# #--------------------------------------------------step 2: preprocessing and labeling for the MI_5min_MidClose window.--------------------------------------------------

# # Select the relevant market impact window
# target_horizon = 'MI_5min_MidClose'

# # Drop rows with missing market impact values for that horizon
# df_clean = df.dropna(subset=[target_horizon, 'Tweet', 'Twitter_acc'])

# # Clean tweet text by removing emojis and hyperlinks
# print("Cleaning tweet text...")
# df_clean['Tweet_Original'] = df_clean['Tweet'].copy()  # Keep original for reference
# df_clean['Tweet'] = df_clean['Tweet'].apply(clean_tweet_text)

# # Remove rows where tweet becomes empty after cleaning
# df_clean = df_clean[df_clean['Tweet'].str.len() > 0]

# # Define three-class classification target with neutral zone
# def classify_market_impact(x):
#     """
#     Classify market impact into three categories:
#     - Buy: movement > 5 basis points (0.05%)
#     - Sell: movement < -5 basis points (-0.05%)  
#     - Neutral: movement between -5 and +5 basis points
#     """
#     if x > 0.0005:  # > 5 basis points
#         return 'Buy'
#     elif x < -0.0005:  # < -5 basis points
#         return 'Sell'
#     else:  # between -5 and +5 basis points
#         return 'Neutral'

# df_clean['Label'] = df_clean[target_horizon].apply(classify_market_impact)

# # Display class balance and a sample
# label_counts = df_clean['Label'].value_counts()

# print(f"Dataset size after cleaning: {len(df_clean)} rows")
# print(f"Original dataset size: {len(df)} rows")
# print(f"Rows removed: {len(df) - len(df_clean)}")
# print("\nFirst 10 cleaned tweets:")
# print(df_clean[['Tweet', 'Label']].head(10))
# print("\n-----------")
# print("Class distribution:")
# print(label_counts)

# #----------------------------------------------------- Step 3: Text Vectorisation + Account Encoding-----------------------------------------------------

# # TF-IDF vectorisation on tweets - Enhanced parameters for better performance
# print("Vectorizing text features...")
# tfidf = TfidfVectorizer(
#     max_features=2000,          # Increased from 1000 for richer features
#     stop_words='english',
#     min_df=2,                   # Ignore terms in less than 2 documents
#     max_df=0.95,                # Ignore terms in more than 95% of documents
#     ngram_range=(1, 2),         # Include unigrams and bigrams
#     lowercase=True,
#     strip_accents='unicode'
# )

# X_text = tfidf.fit_transform(df_clean['Tweet'])

# # One-hot encode Twitter account
# print("Encoding Twitter accounts...")
# ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
# X_account = ohe.fit_transform(df_clean[['Twitter_acc']])

# # Combine all features
# print("Combining features...")
# X = hstack([X_text, X_account])

# # Target variable - Convert to numeric labels for multi-class MLP (0=Sell, 1=Neutral, 2=Buy)
# label_mapping = {'Sell': 0, 'Neutral': 1, 'Buy': 2}
# y = df_clean['Label'].map(label_mapping)

# print(f"Label distribution after mapping:")
# for label, code in label_mapping.items():
#     count = (y == code).sum()
#     percentage = (count / len(y)) * 100
#     print(f"  {label} ({code}): {count} ({percentage:.1f}%)")

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# print("-" * 60)
# print(f"Training set shape: {X_train.shape}")
# print(f"Test set shape: {X_test.shape}")
# print(f"Training class distribution:")
# for label, code in label_mapping.items():
#     count = (y_train == code).sum()
#     print(f"  {label}: {count}")
    
# print(f"Test class distribution:")
# for label, code in label_mapping.items():
#     count = (y_test == code).sum()
#     print(f"  {label}: {count}")
# print("-" * 60)

# #----------------------------------------------------- Step 4: Random Forest Training and Evaluation-----------------------------------------------------

# print("Training Random Forest Classifier...")

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# import time

# # Configure Random Forest Classifier with optimized parameters
# rf_model = RandomForestClassifier(
#     n_estimators=500,              # Number of trees (reduced from 1000 for speed)
#     max_depth=15,                  # Maximum depth of trees
#     min_samples_split=5,           # Minimum samples to split a node
#     min_samples_leaf=2,            # Minimum samples in leaf node
#     max_features='sqrt',           # Number of features to consider for splits
#     bootstrap=True,                # Use bootstrap sampling
#     random_state=42,               # For reproducibility
#     n_jobs=-1,                     # Use all available processors
#     verbose=1,                     # Show progress
#     class_weight='balanced'        # Handle class imbalance
# )

# # Train the Random Forest model
# print("Fitting Random Forest model...")
# start_time = time.time()
# rf_model.fit(X_train, y_train)
# training_time = time.time() - start_time

# # Training completion info
# print(f"Training completed in {training_time:.2f} seconds")
# print(f"Number of trees: {rf_model.n_estimators}")
# print(f"Number of features used: {rf_model.n_features_in_}")

# # Make predictions
# print("Making predictions...")
# y_pred_proba = rf_model.predict_proba(X_test)  # Probabilities for all classes
# y_pred = rf_model.predict(X_test)

# # Convert back to original labels for reporting
# reverse_label_mapping = {0: 'Sell', 1: 'Neutral', 2: 'Buy'}
# y_test_labels = [reverse_label_mapping[label] for label in y_test]
# y_pred_labels = [reverse_label_mapping[pred] for pred in y_pred]

# #----------------------------------------------------- Step 5: Random Forest Performance Evaluation-----------------------------------------------------

# print("\n" + "="*70)
# print("RANDOM FOREST PERFORMANCE EVALUATION")
# print("="*70)

# # Basic accuracy
# accuracy = accuracy_score(y_test_labels, y_pred_labels)
# print(f"Test Accuracy: {accuracy:.4f}")

# # Training accuracy for comparison
# train_accuracy = rf_model.score(X_train, y_train)
# print(f"Training Accuracy: {train_accuracy:.4f}")
# print(f"Generalization Gap: {abs(train_accuracy - accuracy):.4f}")

# if abs(train_accuracy - accuracy) > 0.1:
#     print("⚠️  Large gap suggests possible overfitting")
# elif abs(train_accuracy - accuracy) > 0.05:
#     print("⚠️  Moderate overfitting detected")
# else:
#     print("✅ Good generalization performance")

# # ROC AUC Score - For multi-class, use macro or weighted average
# try:
#     roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
#     print(f"Multi-class ROC AUC Score (weighted): {roc_auc:.4f}")
# except Exception as e:
#     print(f"ROC AUC: Could not calculate - {e}")

# # Out-of-bag score (unique to Random Forest)
# if hasattr(rf_model, 'oob_score_'):
#     print(f"Out-of-bag Score: {rf_model.oob_score_:.4f}")
# else:
#     print("Out-of-bag Score: Not available (oob_score=False)")

# # Detailed classification report
# print("\nDetailed Classification Report:")
# print("-" * 50)
# report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
# print(classification_report(y_test_labels, y_pred_labels))

# # Confusion Matrix Analysis for 3 classes
# conf_matrix = confusion_matrix(y_test_labels, y_pred_labels, labels=['Sell', 'Neutral', 'Buy'])
# print("\nConfusion Matrix (3x3):")
# print("-" * 40)
# print("                   Predicted")
# print("              Sell  Neutral  Buy")
# print(f"Actual Sell   {conf_matrix[0,0]:4d}    {conf_matrix[0,1]:4d}   {conf_matrix[0,2]:4d}")
# print(f"       Neutral{conf_matrix[1,0]:4d}    {conf_matrix[1,1]:4d}   {conf_matrix[1,2]:4d}")
# print(f"       Buy    {conf_matrix[2,0]:4d}    {conf_matrix[2,1]:4d}   {conf_matrix[2,2]:4d}")

# # Calculate class-specific metrics
# print(f"\nDetailed Class Performance:")
# for i, class_name in enumerate(['Sell', 'Neutral', 'Buy']):
#     tp = conf_matrix[i, i]  # True positives for this class
#     fp = conf_matrix[:, i].sum() - tp  # False positives
#     fn = conf_matrix[i, :].sum() - tp  # False negatives
    
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#     f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
#     print(f"{class_name:8s}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

# #----------------------------------------------------- Step 6: Cross-Validation Analysis-----------------------------------------------------

# print("\n" + "="*50)
# print("CROSS-VALIDATION ANALYSIS")
# print("="*50)

# # Perform 5-fold stratified cross-validation
# cv_scores = cross_val_score(rf_model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy')

# print(f"5-Fold Cross-Validation Scores: {[f'{score:.4f}' for score in cv_scores]}")
# print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
# print(f"CV Standard Deviation: {cv_scores.std():.4f}")

# if cv_scores.std() < 0.05:
#     print("✅ Low variance - Model is stable across folds")
# else:
#     print("⚠️  High variance - Model performance varies across folds")

# #----------------------------------------------------- Step 7: Random Forest Feature Importance Analysis-----------------------------------------------------

# print("\n" + "="*50)
# print("RANDOM FOREST FEATURE IMPORTANCE ANALYSIS")
# print("="*50)

# # Get feature names
# tfidf_features = tfidf.get_feature_names_out()
# account_features = [f"account_{cat}" for cat in ohe.categories_[0]]
# all_features = list(tfidf_features) + account_features

# print(f"Total features used: {len(all_features)}")
# print(f"Text features (TF-IDF): {len(tfidf_features)}")
# print(f"Account features: {len(account_features)}")

# # Random Forest Feature Importance (unique advantage of RF)
# feature_importance = rf_model.feature_importances_
# top_features_idx = np.argsort(feature_importance)[-15:][::-1]

# print(f"\nTop 15 Most Important Features (Random Forest):")
# print("-" * 60)
# for i, idx in enumerate(top_features_idx):
#     feature_name = all_features[idx] if idx < len(all_features) else f"feature_{idx}"
#     importance = feature_importance[idx]
#     feature_type = "Text" if idx < len(tfidf_features) else "Account"
#     print(f"{i+1:2d}. {feature_name:20s} | {importance:.6f} | {feature_type}")

# # Tree depth statistics
# tree_depths = [tree.tree_.max_depth for tree in rf_model.estimators_]
# print(f"\nRandom Forest Tree Statistics:")
# print(f"Average tree depth: {np.mean(tree_depths):.2f}")
# print(f"Max tree depth: {np.max(tree_depths)}")
# print(f"Min tree depth: {np.min(tree_depths)}")

# # Feature importance by category
# text_importance = np.sum(feature_importance[:len(tfidf_features)])
# account_importance = np.sum(feature_importance[len(tfidf_features):])

# print(f"\nFeature Importance by Category:")
# print(f"Text features total importance: {text_importance:.4f} ({text_importance*100:.1f}%)")
# print(f"Account features total importance: {account_importance:.4f} ({account_importance*100:.1f}%)")

# #----------------------------------------------------- Step 8: Model Comparison Metrics-----------------------------------------------------

# print("\n" + "="*50)
# print("RANDOM FOREST MODEL SUMMARY")
# print("="*50)

# # Model complexity metrics
# print(f"Model Complexity:")
# print(f"• Number of trees: {rf_model.n_estimators}")
# print(f"• Average tree depth: {np.mean(tree_depths):.1f}")
# print(f"• Total parameters (approx): {rf_model.n_estimators * np.mean(tree_depths) * len(all_features):.0f}")

# # Training efficiency
# print(f"\nTraining Efficiency:")
# print(f"• Training time: {training_time:.2f} seconds")
# print(f"• Features processed: {len(all_features):,}")
# print(f"• Training samples: {len(X_train):,}")

# # Memory efficiency (Random Forest advantage)
# print(f"\nRandom Forest Advantages:")
# print("✅ Handles sparse features efficiently")
# print("✅ Provides feature importance rankings")
# print("✅ Robust to outliers")
# print("✅ No feature scaling required")
# print("✅ Built-in cross-validation (out-of-bag)")

# print("\n" + "="*70)
# print("RANDOM FOREST TRAINING COMPLETED SUCCESSFULLY!")
# print("="*70)
# print("Key Results:")
# print(f"• Test Accuracy: {accuracy:.4f}")
# print(f"• ROC AUC: {roc_auc:.4f}" if 'roc_auc' in locals() else "• ROC AUC: N/A")
# print(f"• CV Mean Accuracy: {cv_scores.mean():.4f}")
# print(f"• Training Time: {training_time:.2f} seconds")
# print(f"• Most Important Feature: {all_features[top_features_idx[0]]}")
# print("="*70)


import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
file_path = "tweet_market_impact.xlsx"
xls = pd.ExcelFile(file_path)
df = xls.parse('Sheet1')

# Text cleaning utilities
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

# Define horizons
horizons = [
    'MI_1min_MidClose', 'MI_5min_MidClose', 'MI_15min_MidClose',
    'MI_30min_MidClose', 'MI_1h_MidClose', 'MI_2h_MidClose',
    'MI_4h_MidClose', 'MI_8h_MidClose', 'MI_12h_MidClose',
    'MI_1d_MidClose', 'MI_2d_MidClose'
]

performance_summary = []

# Main loop for each horizon
for horizon in horizons:
    df_clean = df.dropna(subset=[horizon, 'Tweet', 'Twitter_acc']).copy()
    df_clean['Tweet'] = df_clean['Tweet'].apply(clean_tweet_text)
    df_clean = df_clean[df_clean['Tweet'].str.len() > 0]

    def classify_market_impact(x):
        return 'Buy' if x > 0.0005 else ('Sell' if x < -0.0005 else 'Neutral')

    df_clean['Label'] = df_clean[horizon].apply(classify_market_impact)
    label_mapping = {'Sell': 0, 'Neutral': 1, 'Buy': 2}
    df_clean['Label'] = df_clean['Label'].map(label_mapping)

    if df_clean['Label'].nunique() < 3 or df_clean.shape[0] < 200:
        continue

    tfidf = TfidfVectorizer(max_features=2000, stop_words='english', min_df=2, max_df=0.95, ngram_range=(1, 2))
    X_text = tfidf.fit_transform(df_clean['Tweet'])
    ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    X_account = ohe.fit_transform(df_clean[['Twitter_acc']])
    X = hstack([X_text, X_account])
    y = df_clean['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=5,
        min_samples_leaf=2, max_features='sqrt', bootstrap=True,
        random_state=42, n_jobs=-1, class_weight='balanced'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    inv_map = {0: 'Sell', 1: 'Neutral', 2: 'Buy'}
    y_test_labels = y_test.map(inv_map)
    y_pred_labels = pd.Series(y_pred).map(inv_map)
    report = classification_report(y_test_labels, y_pred_labels, output_dict=True)

    performance_summary.append({
        'Horizon': horizon,
        'Accuracy': accuracy_score(y_test_labels, y_pred_labels),
        'Precision_Buy': report.get('Buy', {}).get('precision', 0),
        'Precision_Sell': report.get('Sell', {}).get('precision', 0),
        'F1_Buy': report.get('Buy', {}).get('f1-score', 0),
        'F1_Sell': report.get('Sell', {}).get('f1-score', 0)
    })

performance_summary_df = pd.DataFrame(performance_summary)

performance_summary_df.to_csv("RF_performance_summary.csv", index=False)

