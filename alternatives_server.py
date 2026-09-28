import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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
    sugar: float = Field(..., ge=0)
    salt: float = Field(..., ge=0)
    fat: float = Field(..., ge=0)
    preservative: float = Field(..., ge=0)
    fiber: float = Field(..., ge=0)
    protein: float = Field(..., ge=0)
    improvement_goal: str


@app.post("/alternatives")
async def alternatives(f: Formulation):
    goal_label = {
        "health": "improve nutritional profile / health impact",
        "shelf_life": "extend shelf life and stability",
        "cost": "reduce ingredient cost while maintaining quality",
    }.get(f.improvement_goal, f.improvement_goal)

    prompt = f"""You are a senior food scientist AI specializing in extruded snack formulations.

CURRENT FORMULATION:
- Sugar: {f.sugar} g
- Salt: {f.salt} g
- Fat: {f.fat} g
- Preservative: {f.preservative} g
- Fiber: {f.fiber} g
- Protein: {f.protein} g

IMPROVEMENT GOAL: {goal_label}

TASK: Recommend intelligent ingredient substitutions based on real food science.
Consider: stevia, erythritol, allulose, monk fruit, avocado oil, MCT oil, high-oleic sunflower oil,
rosemary extract, tocopherols, citric acid, potassium sorbate, inulin, psyllium, oat bran,
pea protein, whey isolate, chickpea flour, potassium chloride, seaweed, yeast extract.

Return ONLY valid JSON in EXACTLY this shape (no markdown, no code fences):
{{
  "substitutions": [
    {{
      "category": "<Sugar|Fat|Preservative|Protein|Fiber|Salt>",
      "original_ingredient": "<string>",
      "alternative": "<string>",
      "reason": "<1-2 sentences, scientific reason>",
      "health_impact": "<e.g. +25% or Neutral or -10%>",
      "shelf_life_impact": "<e.g. +15% or Neutral or -5%>",
      "cost_impact": "<e.g. +10% or Neutral or -8%>",
      "sensory_impact": "<short phrase, e.g. Slight sweetness drop, crispness maintained>"
    }}
  ],
  "optimized_formulation": {{
    "sugar": <number>,
    "salt": <number>,
    "fat": <number>,
    "preservative": <number>,
    "fiber": <number>,
    "protein": <number>
  }},
  "scores": {{
    "health": <0-100>,
    "shelf_life": <0-100>,
    "cost": <0-100>,
    "sensory": <0-100>
  }},
  "cost_analysis": {{
    "current_cost": <number>,
    "optimized_cost": <number>,
    "savings_percent": <number, positive means savings>
  }},
  "sensory_impact": {{
    "taste": "<Improved|Neutral|Reduced> - <short explanation>",
    "texture": "<Improved|Neutral|Reduced> - <short explanation>",
    "appearance": "<Improved|Neutral|Reduced> - <short explanation>",
    "overall": "<Improved|Neutral|Reduced> - <short explanation>"
  }},
  "scientist_report": "<4-6 sentence professional report: what's wrong with current formulation, which ingredients limit performance, expected benefits after optimization, potential trade-offs, commercial feasibility>",
  "summary": "<2-3 sentence executive summary>"
}}

Provide 3 to 6 substitutions."""

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.4,
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")
