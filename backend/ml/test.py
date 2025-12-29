import joblib
model = joblib.load('backend/ml/wildfire_spread_classifier_advanced.joblib')
print(model.__class__.__module__)