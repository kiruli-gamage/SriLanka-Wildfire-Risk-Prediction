# =============================
# Step 0: Install required packages
# =============================
# pip install pandas numpy scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import PartialDependenceDisplay

# =============================
# Step 1: Load and preprocess MODIS dataset
# =============================
modis_df = pd.read_csv(r"D:/Academics/Machine Learning/Wildfire forecast resarch/Modis/modis_5years_merged.csv.csv")

modis_df['acq_date'] = pd.to_datetime(modis_df['acq_date'], errors='coerce')
modis_df = modis_df.dropna(subset=['latitude', 'longitude'])

# Extract temporal features
modis_df['month'] = modis_df['acq_date'].dt.month
modis_df['year'] = modis_df['acq_date'].dt.year
modis_df['day_of_year'] = modis_df['acq_date'].dt.dayofyear

# Binary target variable
modis_df['fire_occurrence'] = modis_df['confidence'].apply(lambda x: 1 if x > 50 else 0)
modis_df['daynight'] = modis_df['daynight'].map({'D': 0, 'N': 1})

# =============================
# Step 2: Basic Data Insights
# =============================
plt.figure(figsize=(8, 5))
sns.countplot(x='fire_occurrence', data=modis_df, palette='coolwarm')
plt.title('🔥 Fire Occurrence Distribution')
plt.xlabel('Fire Occurrence (0 = No Fire, 1 = Fire)')
plt.ylabel('Count')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(modis_df[['brightness', 'bright_t31', 'frp', 'scan', 'track', 'fire_occurrence']].corr(), annot=True, cmap='RdYlBu_r')
plt.title('Feature Correlation Heatmap')
plt.show()

# =============================
# Step 3: Define features and split data
# =============================
features = [
    'latitude', 'longitude', 'brightness', 'bright_t31', 'frp',
    'scan', 'track', 'daynight', 'month', 'year', 'day_of_year'
]
X = modis_df[features].fillna(0)
y = modis_df['fire_occurrence']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# Step 4: Train Multiple Models
# =============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
}

results = []
print("\n================= Model Results =================\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append((name, acc, f1))
    print(f"{name} → Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

# =============================
# Step 5: Model Comparison
# =============================
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1_Score'])

plt.figure(figsize=(10, 5))
sns.barplot(x='Model', y='Accuracy', data=results_df, palette='Blues_d')
plt.title('🔍 Model Accuracy Comparison')
plt.xticks(rotation=25)
plt.ylim(0, 1)
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x='Model', y='F1_Score', data=results_df, palette='Oranges_d')
plt.title('🧠 Model F1 Score Comparison')
plt.xticks(rotation=25)
plt.ylim(0, 1)
plt.show()

best_model_name = results_df.sort_values(by='F1_Score', ascending=False).iloc[0]['Model']
best_model = models[best_model_name]
print(f"\n✅ Best Model Selected: {best_model_name}")

# =============================
# Step 6: Explainable Visualizations (XAI)
# =============================

# --- 6.1 Confusion Matrix ---
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# --- 6.2 ROC Curve ---
y_prob_best = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob_best)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title(f'ROC Curve - {best_model_name}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# --- 6.3 Precision-Recall Curve ---
precision, recall, _ = precision_recall_curve(y_test, y_prob_best)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color='green', lw=2)
plt.title(f'Precision-Recall Curve - {best_model_name}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

# --- 6.4 Feature Importance (for tree-based models) ---
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='coolwarm')
    plt.title(f'🎯 Feature Importance - {best_model_name}')
    plt.show()

# --- 6.5 Partial Dependence Plots ---
if hasattr(best_model, "feature_importances_"):
    top_features = feature_importance_df['Feature'].head(3).tolist()
    PartialDependenceDisplay.from_estimator(best_model, X_train, features=top_features, feature_names=features)
    plt.suptitle(f'📈 Partial Dependence Plots ({best_model_name})', y=1.02)
    plt.show()

print("\n✅ All visualizations generated successfully!")
