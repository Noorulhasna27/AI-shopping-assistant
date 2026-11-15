import json

from src.agent.executor import run
from src.agent.schemas import Plan


def P(**kw):
    base = dict(intent="compare", product_mentions=[], aspects=[], max_price=None,
                brand=None, use_case=None, clarification_question=None)
    base.update(kw)
    return Plan(**base)


TESTS = [
    P(intent="recommend", max_price=1000, brand="Lenovo", use_case="gaming performance"),
    P(intent="compare", product_mentions=["Acer Nitro 5", "Legion 5"], aspects=["battery"]),
    P(intent="alternatives", product_mentions=["Razer Blade 15"], max_price=1500),
    P(intent="reviews", product_mentions=["Hoverboard"]),
]

for p in TESTS:
    ev = run(p, {})
    print("\n==", p.intent, p.product_mentions, "->", list(ev.keys()))
    print(json.dumps(ev, default=str)[:500])
