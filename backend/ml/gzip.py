import pickletools

with open("backend/ml/wildfire_spread_classifier_advanced.joblib", "rb") as f:
    data = f.read()

print(pickletools.dis(data))