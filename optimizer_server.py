import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv
import json

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
    corn_flour: float = Field(..., ge=0)
    rice_flour: float = Field(..., ge=0)
    chickpea_flour: float = Field(..., ge=0)
    pea_protein: float = Field(..., ge=0)
    oat_fiber: float = Field(..., ge=0)
    oil: float = Field(..., ge=0)
    salt: float = Field(..., ge=0)
    goal: str
    target_protein: float = Field(..., ge=0)
    target_fiber: float = Field(..., ge=0)
    max_fat: float = Field(..., ge=0)
    max_cost: float = Field(..., ge=0)


@app.post("/optimize_formulation")
async def optimize_formulation(f: Formulation):
    prompt = f"""You are a senior food scientist AI specializing in extruded snack and flour-based product formulation.

CURRENT FORMULATION (percent by weight):
- Corn Flour: {f.corn_flour} %
- Rice Flour: {f.rice_flour} %
- Chickpea Flour: {f.chickpea_flour} %
- Pea Protein: {f.pea_protein} %
- Oat Fiber: {f.oat_fiber} %
- Oil: {f.oil} %
- Salt: {f.salt} %

OPTIMIZATION GOAL: {f.goal}

TARGETS:
- Target Protein: {f.target_protein} %
- Target Fiber: {f.target_fiber} %
- Max Fat: {f.max_fat} %
- Max Cost: {f.max_cost} INR/kg

TASK:
1. Produce an optimized formulation that satisfies the targets and the goal as closely as possible.
2. The optimized formulation percentages MUST sum to 100.
3. Provide current and optimized nutrition estimates (protein %, fiber %, fat %, calories kcal/100g).
4. Provide current and optimized cost index in INR/kg (relative, realistic for Indian market).
5. Score the optimized formulation 0-100 based on how well it meets targets and goal.

Return ONLY valid JSON in EXACTLY this shape (no markdown, no code fences):
{{
  "optimized_formulation": {{
    "corn_flour": <number>,
    "rice_flour": <number>,
    "chickpea_flour": <number>,
    "pea_protein": <number>,
    "oat_fiber": <number>,
    "oil": <number>,
    "salt": <number>
  }},
  "nutrition_comparison": {{
    "current": {{ "protein": <number>, "fiber": <number>, "fat": <number>, "calories": <number> }},
    "optimized": {{ "protein": <number>, "fiber": <number>, "fat": <number>, "calories": <number> }}
  }},
  "cost_analysis": {{
    "current_cost": <number>,
    "optimized_cost": <number>,
    "savings_percent": <number>
  }},
  "formulation_score": <0-100>,
  "scientist_report": "<A professional 5-8 sentence report covering: what was wrong with the current formulation, what was changed and why, expected nutritional improvements, trade-offs, and commercial feasibility>"
}}"""

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.35,
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")
