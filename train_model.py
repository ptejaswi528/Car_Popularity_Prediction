import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# 1. Load Dataset
file_path = 'car_evaluation.csv'

if not os.path.exists(file_path):
    print(f"ERROR: {file_path} not found in folder!")
else:
    # Adding names manually because the UCI file usually lacks a header row
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'popularity']
    df = pd.read_csv(file_path, names=columns)

    print(f"Dataset loaded. Rows found: {len(df)}")

    if len(df) == 0:
        print("ERROR: The CSV file is empty. Please check the file content.")
    else:
        # 2. Encoding
        le = LabelEncoder()
        for col in df.columns:
            df[col] = le.fit_transform(df[col])

        X = df.drop('popularity', axis=1)
        y = df['popularity']

        # 3. Train XGBoost Model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
        model.fit(X_train, y_train)

        # 4. Save Model
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)

        print(f"Success! Model trained and saved. Test Accuracy: {model.score(X_test, y_test):.2%}")
import json

# ... (Previous training code here) ...

# Calculate Feature Importance for the Dashboard
importance = model.feature_importances_
feature_names = ['Price', 'Maint', 'Doors', 'Persons', 'Luggage', 'Safety']
importance_data = dict(zip(feature_names, [round(float(i), 2) for i in importance]))

# Create a summary for the KPIs
summary = {
    "total_records": len(df),
    "accuracy": f"{model.score(X_test, y_test):.2%}",
    "feature_importance": importance_data,
    "top_factors": sorted(importance_data, key=importance_data.get, reverse=True)[:3]
}

with open('data_summary.json', 'w') as f:
    json.dump(summary, f)
# Check if these lines are in your train_model.py
counts = df['popularity'].value_counts().to_dict()
summary = {
    # ... other keys ...
    "class_dist": [int(counts.get('unacc', 0)), int(counts.get('acc', 0)), 
                   int(counts.get('good', 0)), int(counts.get('vgood', 0))],
}
# Check that these lines are in your train_model.py
counts = df['popularity'].value_counts().to_dict()

# The keys 'unacc', 'acc', etc., must match your dataset labels exactly
summary = {
    "total_records": len(df),
    "accuracy": f"{model.score(X_test, y_test):.1%}",
    "feature_importance": importance_data,
    "class_dist": [
        int(counts.get('unacc', 0)), 
        int(counts.get('acc', 0)), 
        int(counts.get('good', 0)), 
        int(counts.get('vgood', 0))
    ],
    "top_factor": sorted(importance_data, key=importance_data.get, reverse=True)[0]
}

with open('data_summary.json', 'w') as f:
    json.dump(summary, f)