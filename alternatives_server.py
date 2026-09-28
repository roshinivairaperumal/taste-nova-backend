import os
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

client = Groq(api_key=os.environ["GROQ_API_KEY"])

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Formulation(BaseModel):
    sugar: float
    salt: float
    fat: float
    preservative: float
    fiber: float
    protein: float
    improvement_goal: str

@app.post("/alternatives")
async def alternatives(f: Formulation):
    prompt = f"""You are a senior food scientist AI. Analyze this extruded snack formulation and recommend intelligent ingredient alternatives.

CURRENT FORMULATION:
- Sugar: {f.sugar} g
- Salt: {f.salt} g
- Fat: {f.fat} g
- Preservative: {f.preservative} g
- Fiber: {f.fiber} g
- Protein: {f.protein} g

IMPROVEMENT GOAL: {f.improvement_goal}

Return ONLY valid JSON in EXACTLY this shape (no markdown, no code fences):
{{
  "suggestions": [
    {{
      "original_ingredient": "<string>",
      "alternative": "<string>",
      "reasoning": "<string>",
      "health_impact": "<e.g. +25% or Neutral or -10%>",
      "shelf_life_impact": "<e.g. +15% or Neutral or -5%>",
      "cost_impact": "<e.g. +10% or Neutral or -8%>",
      "confidence": "<e.g. 92%>"
    }}
  ],
  "explanation": {{
    "why": "<string, 2-3 sentences explaining the overall recommendation strategy>",
    "trade_offs": ["<string>", "<string>"],
    "risks": ["<string>", "<string>"],
    "benefits": ["<string>", "<string>"]
  }},
  "optimization_summary": {{
    "health_improvement": "<e.g. +30%>",
    "shelf_life_improvement": "<e.g. +20%>",
    "cost_change": "<e.g. +5%>"
  }}
}}

Provide 3 to 5 suggestions. Base recommendations on real food science:
- Sugar alternatives: stevia, erythritol, allulose, monk fruit, maltitol
- Fat alternatives: avocado oil, MCT oil, high-oleic sunflower oil, inulin
- Preservative alternatives: rosemary extract, tocopherols, citric acid, potassium sorbate
- Fiber boosters: inulin, psyllium, oat bran, beta-glucan
- Protein sources: pea protein, whey isolate, soy protein, chickpea flour
- Salt reduction: potassium chloride, seaweed, yeast extract
Tailor suggestions to the improvement goal ({f.improvement_goal})."""

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    return json.loads(completion.choices[0].message.content)
