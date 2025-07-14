import os
import pandas as pd
from datetime import datetime

# Define the folder containing the Excel files
folder_path = "downloaded_tweets"

# List to store all DataFrames
all_dfs = []

# Loop through all Excel files in the folder
for file in os.listdir(folder_path):
    if file.endswith(".xlsx"):
        # Full path to the file
        file_path = os.path.join(folder_path, file)

        # Read the Excel file
        df = pd.read_excel(file_path)

        # Extract Twitter handle from filename (remove extension)
        twitter_handle = os.path.splitext(file)[0]

        # Add new column
        df["Twitter_acc"] = twitter_handle

        # Convert timestamp to ISO 8601 format if the column exists
        if "timestamp" in df.columns:
            df["timestamp"] = df["timestamp"].apply(
                lambda x: datetime.strptime(x, "%b %d, %Y · %I:%M %p UTC").isoformat() + "Z"
                if isinstance(x, str) else x
            )

        # Append to list
        all_dfs.append(df)

# Concatenate all DataFrames
merged_df = pd.concat(all_dfs, ignore_index=True)

# Save the merged DataFrame to a new Excel file
merged_df.to_excel("merged_tweets_new.xlsx", index=False)

print("Merging complete. File saved as 'merged_tweets.xlsx'.")