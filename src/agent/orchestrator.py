import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.agent.schemas import Plan

load_dotenv()
client = genai.Client()
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gemini-3.5-flash-lite")

SYSTEM = """You are the planner for a laptop shopping assistant.
Read the user's message and return a JSON plan. Do not answer the question.

intent (pick exactly one):
- compare: the user wants two or more named laptops compared
- reviews: the user wants to know what reviewers say about a laptop or an aspect
- alternatives: the user wants similar, cheaper or better options than a named laptop
- recommend: the user wants suggestions from a budget or use case, with no laptop named
- clarify: a key detail is missing or the message is too vague to act on
- off_topic: the message is not about shopping for laptops

Rules:
- product_mentions: laptop names exactly as the user wrote them. Never invent names.
- aspects: short words such as battery, display, keyboard, fans, noise, performance, build.
- All catalog prices are in US dollars. max_price is a number in USD.
  If the user gives a budget in another currency, use intent clarify and ask
  for the budget in USD.
- Use saved preferences (budget, brand, use case) when the message does not
  override them.
- Unused fields: empty list or null.
- clarification_question: set only when intent is clarify."""


def make_plan(question, memory=None):
    prefs = memory if memory else "none"
    resp = client.models.generate_content(
        model=ROUTER_MODEL,
        contents=f"Saved preferences: {prefs}\nUser message: {question}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            response_mime_type="application/json",
            response_schema=Plan,
            temperature=0,
        ),
    )
    return Plan.model_validate_json(resp.text)
