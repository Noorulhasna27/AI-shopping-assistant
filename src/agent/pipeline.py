import time

from src.agent.answer import check_grounding, write_answer
from src.agent.executor import run
from src.agent.memory import update_memory
from src.agent.orchestrator import make_plan

REFUSAL = ("I can only help with shopping for laptops: comparing models, "
           "reading what reviewers say, and finding alternatives. "
           "What would you like to know?")
BUSY = "The AI service is busy right now. Please try again in a few seconds."
QUOTA = ("I've hit the free AI usage limit for the moment. Please try again in a few "
         "minutes. Sample questions you've already asked still work instantly.")
USES_PREFS = ("recommend", "alternatives", "compare")


def _retry(fn, *args, tries=2):
    for i in range(tries):
        try:
            return fn(*args)
        except Exception as e:
            print("LLM error:", repr(e)[:300])
            if i == tries - 1 or "RESOURCE_EXHAUSTED" in str(e):
                raise
            time.sleep(3)


def _busy(e):
    return QUOTA if ("RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)) else BUSY


def handle(question, memory, chosen=None, plan=None):
    try:
        plan = plan or _retry(make_plan, question, memory)
    except Exception as e:
        return {"type": "text", "text": _busy(e), "plan": None}
    update_memory(memory, plan)
    if plan.intent == "off_topic":
        return {"type": "text", "text": REFUSAL, "plan": plan}
    if plan.intent == "clarify":
        text = plan.clarification_question or "Could you tell me a bit more about what you need?"
        return {"type": "text", "text": text, "plan": plan}
    ev = run(plan, memory, chosen)
    if "needs_choice" in ev:
        return {"type": "choice", "options": ev["needs_choice"], "plan": plan}
    if "error" in ev:
        return {"type": "text", "text": ev["error"], "plan": plan}
    prefs = memory if plan.intent in USES_PREFS else {}
    try:
        raw = _retry(write_answer, question, ev, prefs)
        text, issues = check_grounding(raw, ev, prefs)
    except Exception as e:
        return {"type": "text", "text": _busy(e), "plan": plan}
    return {"type": "answer", "text": text, "evidence": ev, "issues": issues, "plan": plan}