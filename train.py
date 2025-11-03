"""
Trains a linear regression model to predict coffee ratings based on price per 100g.
Loads data from a remote CSV, trains the model, and saves the trained model as a pickle file.
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load the coffee analysis data from the URL
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(url)

# Prepare features and target
X = df[["100g_USD"]]
y = df["rating"]

# Train linear regression model
model = LinearRegression()
model.fit(X, y)

# Save the trained model as a pickle file
with open("model_1.pickle", "wb") as f:
    pickle.dump(model, f)
