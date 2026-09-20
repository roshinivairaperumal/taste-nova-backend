import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

print("=" * 50)
print("🤖 TRAINING CHIPS AI MODELS")
print("=" * 50)

# Create models folder
os.makedirs('models', exist_ok=True)

# Load data
df = pd.read_csv('data.csv')
print(f"\n📊 Loaded {len(df)} chip samples")

# ============= SENSORY MODEL =============
print("\n🔮 Training Sensory Prediction Model...")

feature_cols = ['Sugar', 'Salt', 'Fat', 'Preservative', 'Fiber', 'Protein']
target_cols = ['Taste', 'Texture', 'Aroma', 'Appearance', 'Overall', 'CostIndex']

X = df[feature_cols]
y = df[target_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_sensory = StandardScaler()
X_train_scaled = scaler_sensory.fit_transform(X_train)
X_test_scaled = scaler_sensory.transform(X_test)

sensory_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
sensory_model.fit(X_train_scaled, y_train)

y_pred = sensory_model.predict(X_test_scaled)
print("\n📈 Sensory Model Performance:")
for i, col in enumerate(target_cols):
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    print(f"  {col}: MAE = {mae:.3f}")

# ============= SHELF LIFE MODEL =============
print("\n⏰ Training Shelf Life Prediction Model...")

le = LabelEncoder()
df['PackagingEncoded'] = le.fit_transform(df['PackagingType'])

shelf_features = ['Sugar', 'Salt', 'Fat', 'Preservative', 'Fiber', 'Protein', 
                  'Moisture', 'StorageTemp', 'PackagingEncoded']
X_shelf = df[shelf_features]
y_shelf = df['ShelfLife']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_shelf, y_shelf, test_size=0.2, random_state=42)

scaler_shelf = StandardScaler()
X_train_s_scaled = scaler_shelf.fit_transform(X_train_s)
X_test_s_scaled = scaler_shelf.transform(X_test_s)

shelf_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
shelf_model.fit(X_train_s_scaled, y_train_s)

y_pred_s = shelf_model.predict(X_test_s_scaled)
mae_shelf = mean_absolute_error(y_test_s, y_pred_s)
print(f"\n📈 Shelf Life Model MAE: {mae_shelf:.2f} days")

# ============= SAVE MODELS =============
print("\n💾 Saving models...")

joblib.dump(sensory_model, 'models/sensory_model.pkl')
joblib.dump(scaler_sensory, 'models/sensory_scaler.pkl')
joblib.dump(shelf_model, 'models/shelf_model.pkl')
joblib.dump(scaler_shelf, 'models/shelf_scaler.pkl')
joblib.dump(le, 'models/packaging_encoder.pkl')

print("\n✅ All models saved to 'models/' folder!")
print("\n🎉 Training complete! Now run: python3 app.py")
