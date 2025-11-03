"""
Trains a Decision Tree Regressor to predict coffee ratings based on price and roast type.
Loads data from a remote CSV, encodes categorical roast values, trains the model, and saves both the model and encoding dictionary as a pickle file.
"""
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import pickle

# Load the coffee analysis data from the URL
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(url)

# Create a dictionary mapping roast categories to numbers
roast_categories = df["roast"].unique()
roast_cat = {cat: idx for idx, cat in enumerate(roast_categories)}

# Map roast column to numerical labels
df["roast_num"] = df["roast"].map(roast_cat)

# Prepare features and target
X = df[["100g_USD", "roast_num"]]
y = df["rating"]

# Train Decision Tree Regressor
model = DecisionTreeRegressor()
model.fit(X, y)

# Save the trained model and roast_cat dictionary as a pickle file
with open("model_2.pickle", "wb") as f:
    pickle.dump({"model": model, "roast_cat": roast_cat}, f)
