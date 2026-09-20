"""
Train ML models using YOUR actual CSV data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🍟 TRAINING CHIPS AI WITH YOUR DATA")
print("=" * 60)

# Create models folder
os.makedirs('models', exist_ok=True)

# ============================================
# LOAD AND PREPARE YOUR DATA
# ============================================

print("\n📂 Loading your CSV files...")

# Try to load your sensory data - you'll need to map your columns
# For now, I'll show you how to use the files you shared

# Load reaction kinetics (for understanding shelf life)
kinetics = pd.read_csv('data/31_Table2_Reaction_Rate_Constants_by_Temperature.csv')
print(f"   ✓ Loaded kinetics data ({len(kinetics)} temp points)")

# Load shelf life data (THIS IS YOUR GOLDEN DATA for shelf life prediction)
shelf_data = pd.read_csv('data/31_Table5_Shelf_Life_by_Temperature.csv')
print(f"   ✓ Loaded shelf life data ({len(shelf_data)} conditions)")

# Load sensory data (for saltiness/bitterness predictions)
sensory1 = pd.read_csv('data/083_Table2a_Study1_Sensory_Liking_Scores.csv')
sensory2 = pd.read_csv('data/083_Table2b_Study2_Sensory_Liking_Scores.csv')
print(f"   ✓ Loaded sensory data ({len(sensory1)} + {len(sensory2)} samples)")

# Load brine formulations (for salt alternatives)
brine1 = pd.read_csv('data/083_Table1a_Study1_Brine_Formulations.csv')
brine2 = pd.read_csv('data/083_Table1b_Study2_Brine_Formulations.csv')
print(f"   ✓ Loaded brine formulation data")

# ============================================
# CREATE TRAINING DATASET FOR SENSORY PREDICTION
# ============================================

print("\n🔮 Building Sensory Prediction Dataset...")

# Extract sensory scores from Study 1
sensory_samples = []

# Study 1 - Different salt concentrations
for idx, row in sensory1.iterrows():
    label = row['Treatment_Label']
    # Extract NaCl percentage from label
    if '100_percent' in label:
        nacl = 5.0
    elif '80_percent' in label:
        nacl = 4.0
    elif '60_percent' in label:
        nacl = 3.0
    elif '40_percent' in label:
        nacl = 2.0
    else:
        nacl = 1.0
    
    sensory_samples.append({
        'Sugar': 2.5,  # Default values - you can adjust
        'Salt': nacl,
        'Fat': 25.0,   # Default
        'Preservative': 0.05,  # Default
        'Fiber': 3.0,  # Default
        'Protein': 6.0,  # Default
        'Taste': row['Overall_Taste_Mean'],
        'Texture': row['Texture_Mean'],
        'Aroma': row['Saltiness_Mean'],  # Using saltiness as aroma proxy
        'Appearance': row['Overall_Liking_Mean'],
        'Overall': row['Overall_Liking_Mean'],
        'CostIndex': 5.0 - (nacl / 10)  # Higher salt = lower cost
    })

# Study 2 - NaCl + KCl mixtures
for idx, row in sensory2.iterrows():
    label = row['Treatment_Label']
    if '100_percent' in label:
        nacl = 5.0
    elif '90_percent' in label:
        nacl = 4.5
    elif '80_percent' in label:
        nacl = 4.0
    elif '70_percent' in label:
        nacl = 3.5
    else:
        nacl = 3.0
    
    sensory_samples.append({
        'Sugar': 2.5,
        'Salt': nacl,
        'Fat': 25.0,
        'Preservative': 0.05,
        'Fiber': 3.0,
        'Protein': 6.0,
        'Taste': row['Overall_Taste_Mean'],
        'Texture': row['Texture_Mean'],
        'Aroma': row['Saltiness_Mean'],
        'Appearance': row['Overall_Liking_Mean'],
        'Overall': row['Overall_Liking_Mean'],
        'CostIndex': 5.0 - (nacl / 10)
    })

sensory_df = pd.DataFrame(sensory_samples)
print(f"   ✅ Created {len(sensory_df)} training samples")

# Define features and targets
feature_cols = ['Sugar', 'Salt', 'Fat', 'Preservative', 'Fiber', 'Protein']
target_cols = ['Taste', 'Texture', 'Aroma', 'Appearance', 'Overall', 'CostIndex']

X_sensory = sensory_df[feature_cols]
y_sensory = sensory_df[target_cols]

# Train sensory model
X_train, X_test, y_train, y_test = train_test_split(X_sensory, y_sensory, test_size=0.2, random_state=42)

scaler_sensory = StandardScaler()
X_train_scaled = scaler_sensory.fit_transform(X_train)
X_test_scaled = scaler_sensory.transform(X_test)

sensory_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
sensory_model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = sensory_model.predict(X_test_scaled)
print(f"\n   📈 Sensory Model Performance:")
for i, col in enumerate(target_cols):
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"      {col}: MAE={mae:.2f}, R²={r2:.2f}")

# ============================================
# SHELF LIFE MODEL (Using your real shelf life data)
# ============================================

print("\n⏰ Building Shelf Life Prediction Model...")

# Use your actual shelf life data from Table5
shelf_features = []
shelf_targets = []

for idx, row in shelf_data.iterrows():
    temp = row['Storage_Temperature_Celsius']
    shelf_25 = row['Shelf_Life_25pct_Rejection_Estimate_days']
    shelf_50 = row['Shelf_Life_50pct_Rejection_Estimate_days']
    
    # Create multiple samples with different ingredient combinations
    # (Using your real shelf life data with typical chip formulations)
    for salt in [1.0, 1.5, 2.0]:
        for fat in [20, 25, 30]:
            shelf_features.append({
                'Sugar': 2.5,
                'Salt': salt,
                'Fat': fat,
                'Preservative': 0.05,
                'Fiber': 3.0,
                'Protein': 6.0,
                'Moisture': 1.5,
                'StorageTemp': temp,
                'PackagingType_Standard': 1,
                'PackagingType_Vacuum': 0,
                'PackagingType_Nitrogen': 0
            })
            shelf_targets.append(shelf_50)

shelf_df = pd.DataFrame(shelf_features)
shelf_df['ShelfLife'] = shelf_targets

shelf_feature_cols = ['Sugar', 'Salt', 'Fat', 'Preservative', 'Fiber', 'Protein', 'Moisture', 'StorageTemp']
X_shelf = shelf_df[shelf_feature_cols]
y_shelf = shelf_df['ShelfLife']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_shelf, y_shelf, test_size=0.2, random_state=42)

scaler_shelf = StandardScaler()
X_train_s_scaled = scaler_shelf.fit_transform(X_train_s)
X_test_s_scaled = scaler_shelf.transform(X_test_s)

shelf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
shelf_model.fit(X_train_s_scaled, y_train_s)

y_pred_s = shelf_model.predict(X_test_s_scaled)
print(f"\n   📈 Shelf Life Model Performance:")
print(f"      MAE: {mean_absolute_error(y_test_s, y_pred_s):.1f} days")
print(f"      R²: {r2_score(y_test_s, y_pred_s):.2f}")

# ============================================
# SAVE MODELS
# ============================================

print("\n💾 Saving models...")

joblib.dump(sensory_model, 'models/sensory_model.pkl')
joblib.dump(scaler_sensory, 'models/sensory_scaler.pkl')
joblib.dump(shelf_model, 'models/shelf_model.pkl')
joblib.dump(scaler_shelf, 'models/shelf_scaler.pkl')

# Save feature names for reference
joblib.dump(feature_cols, 'models/sensory_features.pkl')
joblib.dump(shelf_feature_cols, 'models/shelf_features.pkl')

print("\n✅ Models saved successfully!")
print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETE! Now run: python3 app.py")
print("=" * 60)
