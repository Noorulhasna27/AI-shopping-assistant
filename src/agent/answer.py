import json
import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gemini-3.5-flash")

SYSTEM = """You are a laptop shopping assistant. Answer using ONLY the EVIDENCE JSON.
Rules:
- Never state a spec, price, rating or review claim that is not in the evidence.
  If a value is missing or "not available", say so.
- Prices are in US dollars. Quote prices exactly as given. Do not do arithmetic;
  price differences are already written in the "why" fields.
- When you use a review, paraphrase it briefly and cite it as [R<review_id>]
  using the review_id from the evidence.
- If the evidence has few relevant reviews for an aspect, say so instead of guessing.
- Be concise: a short intro, a compact markdown table when comparing, 2-4 bullets,
  then a one-line verdict tied to what the user asked (budget, use case, aspects).
- Say which saved preferences you applied (budget, brand, use case)."""


def write_answer(question, evidence, memory):
    prompt = (f"User question: {question}\nSaved preferences: {memory or 'none'}\n"
              f"EVIDENCE:\n{json.dumps(evidence, default=str)}")
    resp = client.models.generate_content(
        model=ANSWER_MODEL, contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.2))
    return resp.text


def _numbers(s):
    return {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", s)}


def check_grounding(text, evidence, extra=None):
    """Verify citations and prices in the answer against the evidence."""
    ev = json.dumps(evidence, default=str)
    valid = set(re.findall(r'"review_id": (\d+)', ev))
    issues = []
    bad = {c for c in re.findall(r"\[R(\d+)\]", text) if c not in valid}
    for c in bad:
        text = text.replace(f"[R{c}]", "")
    if bad:
        issues.append(f"removed {len(bad)} invalid citation(s)")
    known = _numbers(ev + json.dumps(extra or {}))
    prices = re.findall(r"\$\s?(\d[\d,]*(?:\.\d+)?)", text)
    unverified = sorted({p for p in prices if float(p.replace(",", "")) not in known})
    if unverified:
        issues.append("unverified prices: " + ", ".join(unverified))
    return text, issues
