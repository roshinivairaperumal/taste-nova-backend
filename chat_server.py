import os
import json
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

client = Groq(api_key=os.environ["GROQ_API_KEY"])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are TASTE NOVA Assistant — an AI helper embedded in the TASTE NOVA platform, a food R&D tool for extruded snack development.

You ONLY answer questions related to:
- Extruded snacks and their formulation
- Food science: sensory evaluation, texture, taste, aroma, appearance, crispiness
- Shelf life, stability, moisture, oxidation, packaging
- Ingredient alternatives (sugar, fat, salt, preservatives, fiber, protein)
- Formulation optimization (protein, fiber, fat, cost targets)
- Food processing: extrusion, barrel temperature, screw speed, feed moisture
- TASTE NOVA platform features: Sensory Prediction, Shelf Life Intelligence, AI Ingredient Alternatives, Formulation Optimizer

STRICT RULES:
1. If the user asks anything OUTSIDE the above scope — politics, coding, general knowledge, relationships, news, jokes, math, movies, health advice unrelated to snacks, anything else — politely refuse with exactly this style:
   "I can only help with TASTE NOVA-related questions about extruded snacks and food formulation. Try asking about formulations, sensory prediction, shelf life, or ingredient alternatives."
2. NEVER discuss other AI models, LLM providers, backend architecture, API keys, or technology stack details. If asked, say you are "the TASTE NOVA Assistant."
3. Keep answers SHORT — under 120 words unless the user explicitly asks for detail.
4. Use simple, professional food-science language — like a food scientist talking to a product developer.
5. When relevant, suggest using a TASTE NOVA module (e.g., "For precise results, try the Formulation Optimizer module.")
6. Never claim to predict exact shelf life in days unless the user has run the Shelf Life module.
7. If unsure, offer to help with a related topic from the list above.

Be friendly, focused, and always bring the user back to extruded snack development."""


class ChatRequest(BaseModel):
    messages: List[dict]  # [{ "role": "user"|"assistant", "content": "..." }, ...]


@app.post("/chat")
async def chat(req: ChatRequest):
    # Keep only the last 10 messages to control token usage
    history = req.messages[-10:] if len(req.messages) > 10 else req.messages

    # Sanitize: only allow user/assistant roles, string content
    clean = []
    for m in history:
        role = m.get("role")
        content = m.get("content", "")
        if role in ("user", "assistant") and isinstance(content, str):
            clean.append({"role": role, "content": content[:2000]})

    if not clean:
        raise HTTPException(status_code=400, detail="No valid messages provided.")

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + clean,
            temperature=0.5,
            max_tokens=260,
        )
        reply = completion.choices[0].message.content.strip()
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")
