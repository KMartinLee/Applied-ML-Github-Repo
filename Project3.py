import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#--------------------------------------------------step 0: Import File.--------------------------------------------------

# Load the uploaded Excel file
file_path = "/Users/kilian_1/Desktop/Education/Bayes_MSc_Energy_Trade_and_Finance/Term_3/Applied-ML-Github-Repo/tweet_market_impact.xlsx"
xls = pd.ExcelFile(file_path)

df = xls.parse('Sheet1')

# Display the first few rows to understand the structure
#print(df.head())

#--------------------------------------------------step 1: preprocessing and labeling for the MI_5min_MidClose window.--------------------------------------------------



# Select the relevant market impact window
target_horizon = 'MI_5min_MidClose'

# Drop rows with missing market impact values for that horizon
df_clean = df.dropna(subset=[target_horizon, 'Tweet', 'Twitter_acc'])

# Define binary classification target: 'Buy' (market up), 'Sell' (market down or unchanged)
df_clean['Label'] = df_clean[target_horizon].apply(lambda x: 'Buy' if x > 0 else 'Sell')

# Display class balance and a sample
label_counts = df_clean['Label'].value_counts()

print(df_clean.head(10))
print("-" * 60)
print(label_counts)

#----------------------------------------------------- Step 2: Text Vectorisation + Account Encoding-----------------------------------------------------


# TF-IDF vectorisation on tweets
"""is creating an instance of TfidfVectorizer from scikit-learn's sklearn.feature_extraction.text module. 
This vectorizer is used to convert a collection of text documents into a matrix of TF-IDF (Term Frequency-Inverse Document Frequency) features."""
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')

X_text = tfidf.fit_transform(df_clean['Tweet']) # means you're applying TF-IDF vectorization to the 'Tweet' column of your DataFrame df_clean. 

# One-hot encode Twitter account
"""creates an instance of OneHotEncoder from sklearn.preprocessing, which is used to convert categorical variables into a one-hot encoded format.
Transforms categorical feature(s) into a one-hot numeric array, where each unique category is represented as a binary column (1 if present, 0 otherwise)."""
ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
X_account = ohe.fit_transform(df_clean[['Twitter_acc']]) #performs one-hot encoding on the 'Twitter_acc' column of a DataFrame df_clean.

# Combine all features
X = hstack([X_text, X_account])

# Target variable
y = df_clean['Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("-" * 60)
print(X_train.shape)
print(X_test.shape)
print(f"Number of values in the training data : {y_train.value_counts()}")
print(f"Number of values in the training data : {y_test.value_counts()}")
print("-" * 60)

#----------------------------------------------------- Step 3: we’ll train a Gradient Boosting classifier (LightGBM), evaluate performance-----------------------------------------------------


# Prepare LightGBM dataset
train_data = lgb.Dataset(X_train, label=(y_train == 'Buy').astype(int)) #Converts X_train (features) and y_train (labels) into a LightGBM Dataset.
test_data = lgb.Dataset(X_test, label=(y_test == 'Buy').astype(int), reference=train_data) #Same for test data.

#  Set model parameters:
params = {
    'objective': 'binary',              # Binary classification
    'metric': 'binary_logloss',        # Log loss (lower is better)
    'verbosity': -1,                   # Suppress logging output
    'boosting_type': 'gbdt',           # Gradient Boosted Decision Trees
    'num_leaves': 31,                  # Max leaf nodes per tree (controls model complexity)
    'learning_rate': 0.05,             # Step size shrinkage
    'feature_fraction': 0.9            # Use 90% of features per tree (adds randomness for generalization)
}

# Trains a LightGBM model using the training data.
gbm_model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100, early_stopping_rounds=10)

# Predict on test set
y_pred_proba = gbm_model.predict(X_test) #Predicts probabilities for the positive class ('Buy' → 1).
y_pred = ['Buy' if p > 0.5 else 'Sell' for p in y_pred_proba]

# Evaluate
"""Generates precision, recall, F1-score for each class.
output_dict=True returns it as a dictionary instead of a string."""
report = classification_report(y_test, y_pred, output_dict=True)

accuracy = accuracy_score(y_test, y_pred) #Computes overall accuracy (correct predictions / total predictions).

conf_matrix = confusion_matrix(y_test, y_pred) #Returns a confusion matrix (true/false positives/negatives).
print("Here are the overall results: ")
print(f"Here is the report for {target_horizon} : {report}")
print(f"Here is the accuary for {target_horizon} : {accuracy}")
print(f"Here is the confusion matrix for {target_horizon}: {conf_matrix}")


