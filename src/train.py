import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

# load the RFM table we saved earlier
data = pd.read_csv("data/rfm_features.csv")

print(data.head())
# inputs the model will learn from
X = data[["Recency", "Frequency"]]

# what we want the model to predict
import numpy as np
y = np.log1p(data["Monetary"])

# split: 80% to learn from, 20% to test on
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
# create the model
model = xgb.XGBRegressor(n_estimators=100, random_state=42)

# train it - this is the "learning" step
model.fit(X_train, y_train)

print("Model training complete.")

# test it on the hidden 20%
predictions = model.predict(X_test)

# check how good the guesses were
mae = mean_absolute_error(np.expm1(y_test), np.expm1(predictions))
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)
model.save_model("model.json")
print("Model saved as model.json")