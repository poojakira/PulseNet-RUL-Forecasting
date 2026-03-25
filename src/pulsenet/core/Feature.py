import joblib

# Load your trained Isolation Forest model
model = joblib.load("isolation_forest_model.joblib")

# Extract feature names
feature_names = model.feature_names_in_

print("Feature names used in the model:")
print(feature_names)
