# =============================
# Step 0: Install required packages
# =============================
# pip install pandas numpy scikit-learn matplotlib seaborn folium

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import folium
from folium.plugins import HeatMap

# =============================
# Step 1: Load and preprocess MODIS dataset
# =============================
modis_df = pd.read_csv(r"D:/Academics/Machine Learning/Wildfire forecast resarch/Modis/modis_5years_merged.csv.csv")

# Convert acquisition date to datetime
modis_df['acq_date'] = pd.to_datetime(modis_df['acq_date'], errors='coerce')

# Drop rows with invalid or missing coordinates
modis_df = modis_df.dropna(subset=['latitude', 'longitude'])

# Extract temporal features
modis_df['month'] = modis_df['acq_date'].dt.month
modis_df['year'] = modis_df['acq_date'].dt.year
modis_df['day_of_year'] = modis_df['acq_date'].dt.dayofyear

# Binary target variable
modis_df['fire_occurrence'] = modis_df['confidence'].apply(lambda x: 1 if x > 50 else 0)

# Encode day/night
modis_df['daynight'] = modis_df['daynight'].map({'D': 0, 'N': 1})

# =============================
# Step 2: Define features and target
# =============================
features = [
    'latitude', 'longitude', 'brightness', 'bright_t31', 'frp',
    'scan', 'track', 'daynight', 'month', 'year', 'day_of_year'
]

X = modis_df[features].fillna(0)
y = modis_df['fire_occurrence']

# Standardize numerical features for certain models (like SVM or Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================
# Step 3: Split dataset
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# Step 4: Train Multiple Models
# =============================

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Support Vector Machine": SVC(kernel='rbf', probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
}

results = []

print("\n================= Model Training Results =================\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{name}:\nAccuracy: {acc:.4f} | F1-Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))
    results.append((name, acc, f1))

# =============================
# Step 5: Pick best model (highest F1)
# =============================
best_model_name, best_acc, best_f1 = sorted(results, key=lambda x: x[2], reverse=True)[0]
best_model = models[best_model_name]

print(f"\n✅ Best Model: {best_model_name}")
print(f"Accuracy: {best_acc:.4f} | F1-Score: {best_f1:.4f}")

# =============================
# Step 6: Predict fire probabilities using best model
# =============================
modis_df['fire_probability'] = best_model.predict_proba(X_scaled)[:, 1]
modis_df['fire_probability_norm'] = modis_df['fire_probability'] / modis_df['fire_probability'].max()

# =============================
# Step 7: Create Folium Heatmap
# =============================
map_center = [7.8731, 80.7718]  # Center of Sri Lanka

wildfire_map = folium.Map(location=map_center, zoom_start=7, tiles='CartoDB positron')

# Optional: sample subset for smoother visualization
sampled_df = modis_df.sample(frac=0.1, random_state=42)
heat_data = sampled_df[['latitude', 'longitude', 'fire_probability_norm']].values.tolist()

HeatMap(
    heat_data,
    radius=12,
    blur=8,
    max_zoom=8,
    min_opacity=0.3,
    gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 0.8: 'red', 1: 'darkred'}
).add_to(wildfire_map)

wildfire_map.save("best_model_wildfire_probability_map.html")
print(f"\n🌍 Heatmap saved using {best_model_name} as 'best_model_wildfire_probability_map.html'.")

# =============================
# Step 8: Save dataset
# =============================
modis_df.to_csv('modis_fire_dataset_with_probabilities.csv', index=False)
print("✅ Dataset with fire probabilities saved.")
