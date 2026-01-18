# Sri Lanka Wildfire Risk Prediction

## Overview
This project predicts wildfire risk in Sri Lanka using NASA MODIS satellite data from 2019 to 2023. Machine learning models like Random Forest and Gradient Boosting were applied to identify high-risk areas. The outputs include risk probability maps and interactive visualizations to help early warning and management.

---

## Dataset
The dataset contains MODIS Active Fire records with:
- Latitude & Longitude
- Brightness, Thermal Band (T31)
- Fire Radiative Power (FRP)
- Day/Night Indicator
- Date and Year
- Confidence level for fire events  

Fire occurrences are labeled as:
- `1` for fire (confidence >50%)
- `0` for non-fire

*Stored in:* `data/MODIS_active_fire_2019-2023.csv`

---

## Scripts
| Script | Description |
|--------|-------------|
| `01_data_preprocessing.py` | Load, clean, normalize, and encode data |
| `02_feature_engineering.py` | Create spatiotemporal and thermal features |
| `03_model_training.py` | Train ML models: Logistic Regression, Decision Tree, SVM, KNN, Random Forest, Gradient Boosting |
| `04_model_evaluation.py` | Evaluate models using Accuracy, F1-Score, Confusion Matrices, ROC/PR curves |
| `05_fire_risk_mapping.py` | Generate risk and probability maps using Folium |

---

## Requirements
Python packages required:

```bash
pip install pandas numpy scikit-learn xgboost folium matplotlib seaborn
