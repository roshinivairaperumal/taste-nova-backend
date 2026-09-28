from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Chips Sensory AI System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
print("Loading AI models...")
sensory_model = joblib.load('models/sensory_model.pkl')
sensory_scaler = joblib.load('models/sensory_scaler.pkl')
shelf_model = joblib.load('models/shelf_model.pkl')
shelf_scaler = joblib.load('models/shelf_scaler.pkl')
sensory_features = joblib.load('models/sensory_features.pkl')
shelf_features = joblib.load('models/shelf_features.pkl')
print("✅ Models loaded!")

# ============= ALTERNATIVE INGREDIENTS DATABASE =============
ALTERNATIVES_DB = {
    'sugar': [
        {'name': 'Stevia', 'ratio': 0.2, 'health_boost': '+30%', 'shelf_impact': '+5%', 'cost': '+25%', 'reason': 'Zero calories, natural sweetener'},
        {'name': 'Erythritol', 'ratio': 0.7, 'health_boost': '+25%', 'shelf_impact': '+3%', 'cost': '+20%', 'reason': 'Sugar alcohol, 95% fewer calories'},
        {'name': 'Monk Fruit', 'ratio': 0.15, 'health_boost': '+35%', 'shelf_impact': '+4%', 'cost': '+40%', 'reason': 'Natural zero-calorie, antioxidant rich'},
        {'name': 'Coconut Sugar', 'ratio': 1.0, 'health_boost': '+15%', 'shelf_impact': '-2%', 'cost': '+10%', 'reason': 'Lower glycemic index, minerals'},
        {'name': 'Allulose', 'ratio': 0.8, 'health_boost': '+28%', 'shelf_impact': '+6%', 'cost': '+45%', 'reason': 'Rare sugar, tastes like sugar'},
    ],
    'salt': [
        {'name': 'Potassium Chloride', 'ratio': 0.8, 'health_boost': '+25%', 'shelf_impact': '0%', 'cost': '+15%', 'reason': '50% less sodium, heart healthy'},
        {'name': 'Sea Salt', 'ratio': 0.9, 'health_boost': '+5%', 'shelf_impact': '+2%', 'cost': '+10%', 'reason': 'Trace minerals, less processed'},
        {'name': 'Himalayan Pink', 'ratio': 0.9, 'health_boost': '+8%', 'shelf_impact': '+2%', 'cost': '+20%', 'reason': 'Rich in minerals, natural color'},
        {'name': 'Miso Powder', 'ratio': 0.8, 'health_boost': '+15%', 'shelf_impact': '-3%', 'cost': '+25%', 'reason': 'Adds umami, probiotic benefits'},
    ],
    'fat': [
        {'name': 'Olive Oil', 'ratio': 0.9, 'health_boost': '+25%', 'shelf_impact': '+5%', 'cost': '+40%', 'reason': 'Heart healthy, antioxidants'},
        {'name': 'Avocado Oil', 'ratio': 0.9, 'health_boost': '+28%', 'shelf_impact': '+10%', 'cost': '+50%', 'reason': 'High smoke point, vitamin E rich'},
        {'name': 'Coconut Oil', 'ratio': 0.8, 'health_boost': '+10%', 'shelf_impact': '+15%', 'cost': '+20%', 'reason': 'MCTs, good for high-heat'},
        {'name': 'Rice Bran Oil', 'ratio': 0.9, 'health_boost': '+15%', 'shelf_impact': '+8%', 'cost': '-5%', 'reason': 'Antioxidant rich, stable'},
        {'name': 'Sunflower Oil', 'ratio': 1.0, 'health_boost': '+5%', 'shelf_impact': '-5%', 'cost': '-15%', 'reason': 'Neutral flavor, cost effective'},
    ],
    'preservative': [
        {'name': 'Rosemary Extract', 'ratio': 1.5, 'health_boost': '+25%', 'shelf_impact': '+20%', 'cost': '+30%', 'reason': 'Natural antioxidant, extends shelf life'},
        {'name': 'Vitamin E', 'ratio': 1.2, 'health_boost': '+20%', 'shelf_impact': '+15%', 'cost': '+25%', 'reason': 'Natural preservative, nutrient'},
        {'name': 'Green Tea Extract', 'ratio': 1.3, 'health_boost': '+30%', 'shelf_impact': '+12%', 'cost': '+28%', 'reason': 'Rich in catechins, antioxidant'},
        {'name': 'Citric Acid', 'ratio': 0.8, 'health_boost': '0%', 'shelf_impact': '+8%', 'cost': '-10%', 'reason': 'Natural, adds tartness'},
    ]
}

# ============= REQUEST MODELS =============

class SensoryInput(BaseModel):
    sugar: float
    salt: float
    fat: float
    preservative: float
    fiber: float
    protein: float

class SensoryResponse(BaseModel):
    taste: float
    texture: float
    aroma: float
    appearance: float
    overall: float
    cost_index: float
    recommendation: str

class ShelfLifeInput(BaseModel):
    sugar: float
    salt: float
    fat: float
    preservative: float
    fiber: float
    protein: float
    moisture: float
    storage_temp: float
    packaging_type: str = "Standard"

class ShelfLifeResponse(BaseModel):
    shelf_life_days: float
    quality_grade: str
    risk_factors: List[str]
    recommendations: List[str]

class AlternativesInput(BaseModel):
    sugar: float
    salt: float
    fat: float
    preservative: float
    fiber: float
    protein: float
    improvement_goal: str = "health"  # health, shelf_life, cost

class Alternative(BaseModel):
    original_ingredient: str
    alternative: str
    replacement_ratio: float
    health_impact: str
    shelf_life_impact: str
    cost_impact: str
    reasoning: str

# ============= API ENDPOINTS =============

@app.get("/")
def root():
    return {"message": "🍟 Chips Sensory AI System", "status": "running"}

@app.post("/predict", response_model=SensoryResponse)
async def predict_alias(data: SensoryInput):
    return await predict_sensory(data)

@app.post("/predict_sensory", response_model=SensoryResponse)
def predict_sensory(data: SensoryInput):
    """Predict sensory scores from ingredient percentages"""
    
    # Prepare input
    features = np.array([[data.sugar, data.salt, data.fat, data.preservative, data.fiber, data.protein]])
    features_scaled = sensory_scaler.transform(features)
    
    # Predict
    prediction = sensory_model.predict(features_scaled)[0]
    prediction = np.clip(prediction, 1, 10)  # Scale 1-10
    
    # Generate recommendation
    overall = prediction[4]
    if overall >= 8:
        recommendation = "🌟 Excellent formulation! Premium quality expected."
    elif overall >= 6.5:
        recommendation = "👍 Good formulation. Minor tweaks could make it excellent."
    elif overall >= 5:
        recommendation = "📊 Average formulation. Consider optimizing ratios."
    else:
        recommendation = "⚠️ Poor formulation. Significant reformulation recommended."
    
    # Add specific advice
    if data.sugar > 4:
        recommendation += " Reduce sugar for better health score."
    if data.salt > 1.8:
        recommendation += " High salt may affect consumer acceptance."
    if data.fat > 30:
        recommendation += " High fat content increases cost and health concerns."
    
    return SensoryResponse(
        taste=round(float(prediction[0]), 1),
        texture=round(float(prediction[1]), 1),
        aroma=round(float(prediction[2]), 1),
        appearance=round(float(prediction[3]), 1),
        overall=round(float(prediction[4]), 1),
        cost_index=round(float(prediction[5]), 1),
        recommendation=recommendation
    )

@app.post("/predict_shelf_life", response_model=ShelfLifeResponse)
def predict_shelf_life(data: ShelfLifeInput):
    """Predict shelf life and suggest improvements"""
    
    # Prepare input
    features = np.array([[
        data.sugar, data.salt, data.fat, data.preservative, 
        data.fiber, data.protein, data.moisture, data.storage_temp
    ]])
    features_scaled = shelf_scaler.transform(features)
    
    # Predict
    shelf_life = shelf_model.predict(features_scaled)[0]
    shelf_life = max(30, min(365, shelf_life))  # Cap between 30-365 days
    
    # Determine grade
    if shelf_life >= 180:
        grade = "A - Excellent"
    elif shelf_life >= 120:
        grade = "B - Good"
    elif shelf_life >= 60:
        grade = "C - Fair"
    else:
        grade = "D - Poor"
    
    # Identify risk factors
    risk_factors = []
    if data.moisture > 2.0:
        risk_factors.append("High moisture content (>2%) promotes microbial growth")
    if data.storage_temp > 30:
        risk_factors.append(f"High storage temperature ({data.storage_temp}°C) accelerates oxidation")
    if data.fat > 28:
        risk_factors.append("High fat content increases rancidity risk")
    if data.preservative < 0.03:
        risk_factors.append("Low preservative level may limit shelf life")
    
    # Generate recommendations
    recommendations = []
    if data.moisture > 1.8:
        recommendations.append("🔹 Reduce moisture to <1.5% to extend shelf life by 30-40 days")
    if data.storage_temp > 25:
        recommendations.append(f"🔹 Lower storage temperature to 20-25°C (+{int((30 - data.storage_temp) * 2)} days)")
    if data.fat > 25:
        recommendations.append("🔹 Consider adding natural antioxidants (rosemary extract) +15-20 days")
    if shelf_life < 90:
        recommendations.append("🔹 Switch to vacuum/nitrogen packaging for +30-50 days")
    
    if not recommendations:
        recommendations.append("✅ Current formulation has good shelf life potential")
    
    return ShelfLifeResponse(
        shelf_life_days=round(shelf_life, 0),
        quality_grade=grade,
        risk_factors=risk_factors,
        recommendations=recommendations[:4]
    )

@app.post("/alternatives", response_model=Dict)
def get_alternatives(data: AlternativesInput):
    """Suggest 6-10 alternative ingredients"""
    
    suggestions = []
    
    # Check each ingredient for improvement opportunities
    if data.sugar > 3.0:
        for alt in ALTERNATIVES_DB['sugar'][:3]:
            suggestions.append({
                'original_ingredient': 'Sugar',
                'alternative': alt['name'],
                'replacement_ratio': alt['ratio'],
                'health_impact': alt['health_boost'],
                'shelf_life_impact': alt['shelf_impact'],
                'cost_impact': alt['cost'],
                'reasoning': alt['reason']
            })
    
    if data.salt > 1.5:
        for alt in ALTERNATIVES_DB['salt'][:2]:
            suggestions.append({
                'original_ingredient': 'Salt',
                'alternative': alt['name'],
                'replacement_ratio': alt['ratio'],
                'health_impact': alt['health_boost'],
                'shelf_life_impact': alt['shelf_impact'],
                'cost_impact': alt['cost'],
                'reasoning': alt['reason']
            })
    
    if data.fat > 25:
        for alt in ALTERNATIVES_DB['fat'][:3]:
            suggestions.append({
                'original_ingredient': 'Fat/Oil',
                'alternative': alt['name'],
                'replacement_ratio': alt['ratio'],
                'health_impact': alt['health_boost'],
                'shelf_life_impact': alt['shelf_impact'],
                'cost_impact': alt['cost'],
                'reasoning': alt['reason']
            })
    
    if data.preservative < 0.05:
        for alt in ALTERNATIVES_DB['preservative'][:2]:
            suggestions.append({
                'original_ingredient': 'Preservative',
                'alternative': alt['name'],
                'replacement_ratio': alt['ratio'],
                'health_impact': alt['health_boost'],
                'shelf_life_impact': alt['shelf_impact'],
                'cost_impact': alt['cost'],
                'reasoning': alt['reason']
            })
    
    # Add general improvements if needed
    if len(suggestions) < 6:
        suggestions.append({
            'original_ingredient': 'Fiber',
            'alternative': 'Inulin (Prebiotic Fiber)',
            'replacement_ratio': 1.2,
            'health_impact': '+25%',
            'shelf_life_impact': '+5%',
            'cost_impact': '+20%',
            'reasoning': 'Improves gut health, adds creaminess'
        })
        suggestions.append({
            'original_ingredient': 'Protein',
            'alternative': 'Pea Protein Isolate',
            'replacement_ratio': 1.0,
            'health_impact': '+15%',
            'shelf_life_impact': '0%',
            'cost_impact': '+15%',
            'reasoning': 'Plant-based, allergen-friendly'
        })
    
    # Sort by improvement goal
    if data.improvement_goal == 'health':
        suggestions.sort(key=lambda x: int(x['health_impact'].replace('+', '').replace('%', '')), reverse=True)
    elif data.improvement_goal == 'shelf_life':
        suggestions.sort(key=lambda x: int(x['shelf_life_impact'].replace('+', '').replace('%', '')), reverse=True)
    elif data.improvement_goal == 'cost':
        suggestions.sort(key=lambda x: x['cost_impact'])
    
    return {"suggestions": suggestions[:10]}

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": True}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print("🍟 Starting Chips AI API Server")
    print("=" * 50)
    print("\n📍 API: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
