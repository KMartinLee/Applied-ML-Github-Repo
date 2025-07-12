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

#--------------------------------------------------step 1: Text Cleaning Functions.--------------------------------------------------
def remove_emojis(text):
    """Remove emojis from text using regex pattern"""
    if pd.isna(text):
        return text
    # Pattern to match emoji characters
    emoji_pattern = re.compile("["
                              u"\U0001F600-\U0001F64F"  # emoticons
                              u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                              u"\U0001F680-\U0001F6FF"  # transport & map
                              u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                              u"\U00002702-\U000027B0"  # dingbats
                              u"\U000024C2-\U0001F251"  # enclosed characters
                              "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_hyperlinks(text):
    """Remove hyperlinks from text"""
    if pd.isna(text):
        return text
    # Remove URLs starting with http, https, or www
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = url_pattern.sub('', text)
    
    # Remove www links
    www_pattern = re.compile(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = www_pattern.sub('', text)
    
    # Remove shortened URLs (like t.co, bit.ly, etc.)
    short_url_pattern = re.compile(r'\b(?:t\.co|bit\.ly|tinyurl\.com|goo\.gl|ow\.ly|short\.link)/\S+')
    text = short_url_pattern.sub('', text)
    
    return text

def clean_tweet_text(text):
    """Clean tweet text by removing emojis and hyperlinks"""
    if pd.isna(text):
        return text
    
    # Remove hyperlinks first
    text = remove_hyperlinks(text)
    
    # Remove emojis
    text = remove_emojis(text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

#--------------------------------------------------step 2: preprocessing and labeling for the MI_5min_MidClose window.--------------------------------------------------



# Select the relevant market impact window
target_horizon = 'MI_5min_MidClose'

# Drop rows with missing market impact values for that horizon
df_clean = df.dropna(subset=[target_horizon, 'Tweet', 'Twitter_acc'])

# Clean tweet text by removing emojis and hyperlinks
print("Cleaning tweet text...")
df_clean['Tweet_Original'] = df_clean['Tweet'].copy()  # Keep original for reference
df_clean['Tweet'] = df_clean['Tweet'].apply(clean_tweet_text)

# Remove rows where tweet becomes empty after cleaning
df_clean = df_clean[df_clean['Tweet'].str.len() > 0]

# Define binary classification target: 'Buy' (market up), 'Sell' (market down or unchanged)
df_clean['Label'] = df_clean[target_horizon].apply(lambda x: 'Buy' if x > 0 else 'Sell')

# Display class balance and a sample
label_counts = df_clean['Label'].value_counts()

print(f"Dataset size after cleaning: {len(df_clean)} rows")
print(f"Original dataset size: {len(df)} rows")
print(f"Rows removed: {len(df) - len(df_clean)}")
print("\nFirst 10 cleaned tweets:")
print(df_clean[['Tweet', 'Label']].head(10))
print("\n-----------")
print("Class distribution:")
print(label_counts)

#----------------------------------------------------- Step 3: Text Vectorisation + Account Encoding-----------------------------------------------------


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

#----------------------------------------------------- Step 4: we’ll train a Gradient Boosting classifier (LightGBM), evaluate performance-----------------------------------------------------


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
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(report, accuracy, conf_matrix)


