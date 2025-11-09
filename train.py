"""
Trains a linear regression model to predict coffee ratings based on price per 100g.
Loads data from a remote CSV, trains the model, and saves the trained model as a pickle file.
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import pickle

# Load the coffee analysis data from the URL
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(url)

# Prepare features and target for model_1 (Linear Regression)
X1 = df[["100g_USD"]]
y = df["rating"]

# Train linear regression model
model1 = LinearRegression()
model1.fit(X1, y)

# Save the trained linear regression model as model_1.pickle
with open("model_1.pickle", "wb") as f:
    pickle.dump(model1, f)

# Prepare features for model_2 (Decision Tree)
# Encode 'roast' as categorical if not numeric
X2 = df[["100g_USD", "roast"]].copy()
if X2["roast"].dtype == object:
    X2["roast"] = X2["roast"].astype('category').cat.codes

# Train decision tree regressor
model2 = DecisionTreeRegressor(random_state=42)
model2.fit(X2, y)

# Save the trained decision tree model as model_2.pickle
with open("model_2.pickle", "wb") as f:
    pickle.dump(model2, f)
