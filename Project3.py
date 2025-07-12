import pandas as pd

#--------------------------------------------------step 0: Import File.--------------------------------------------------

# Load the uploaded Excel file
file_path = "/Users/kilian_1/Desktop/Education/Bayes_MSc_Energy_Trade_and_Finance/Term_3/Applied-ML-Github-Repo/tweet_market_impact.xlsx"
xls = pd.ExcelFile(file_path)

df = xls.parse('Sheet1')

# Display the first few rows to understand the structure
#print(df.head())

#--------------------------------------------------step 1: preprocessing and labeling for the MI_5min_MidClose window.--------------------------------------------------

from sklearn.model_selection import train_test_split

# Select the relevant market impact window
target_horizon = 'MI_5min_MidClose'

# Drop rows with missing market impact values for that horizon
df_clean = df.dropna(subset=[target_horizon, 'Tweet', 'Twitter_acc'])

# Define binary classification target: 'Buy' (market up), 'Sell' (market down or unchanged)
df_clean['Label'] = df_clean[target_horizon].apply(lambda x: 'Buy' if x > 0 else 'Sell')

# Display class balance and a sample
label_counts = df_clean['Label'].value_counts()

print(df_clean.head(10))
print("-----------")
print(label_counts)

