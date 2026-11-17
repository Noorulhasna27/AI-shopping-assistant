
import csv
import os
import random
import re
import sys
import time
from pathlib import Path

RUN_LLM_PARTS = True
ANSWER_QUESTIONS = 3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


from src.agent.executor import run 
from src.agent.schemas import Plan  
from src.tools import shop_tools as T  


def P(**kw):
    base = dict(intent="compare", product_mentions=[], aspects=[], max_price=None,
                brand=None, use_case=None, clarification_question=None)
    base.update(kw)
    return Plan(**base)


# 1. retrieval
ASPECTS = {
    "battery life": ["battery", "charge", "charging", "hours"],
    "fan noise and heat": ["fan", "loud", "noise", "noisy", "hot", "heat", "warm", "thermal"],
    "screen quality": ["screen", "display", "panel", "brightness", "resolution", "color", "colour"],
    "keyboard": ["keyboard", "keys", "typing", "backlit", "trackpad", "touchpad"],
    "gaming performance": ["game", "gaming", "fps", "graphics", "performance", "lag"],
    "build quality": ["build", "plastic", "hinge", "sturdy", "flimsy", "durable", "metal", "solid"],
    "speakers and sound": ["speaker", "sound", "audio", "volume", "bass"],
    "value for money": ["price", "value", "worth", "money", "deal", "expensive"],
}


def eval_retrieval(k=5):
    rows = []
    for aspect, words in ASPECTS.items():
        got = T.search_reviews(aspect, k=k, min_score=0.0)["matches"]
        hits = sum(any(w in m["text"].lower() for w in words) for m in got)
        rows.append((aspect, hits, len(got)))
    total = sum(n for _, _, n in rows) or 1
    precision = sum(h for _, h, _ in rows) / total
    hit_rate = sum(1 for _, h, _ in rows if h > 0) / len(rows)
    return rows, precision, hit_rate


# 2. tool correctness
def eval_constraints():
    prod = T._data()[0]
    checks = []

    def check(name, ok):
        checks.append((name, bool(ok)))

    for budget in (800, 1000, 1500):
        ev = run(P(intent="recommend", max_price=budget), {})
        prices = list(ev.get("comparison", {}).get("specs", {}).get("price", {}).values())
        check(f"recommend under ${budget}: every result within budget",
              prices and all(p <= budget for p in prices))
    for brand in prod["brand"].value_counts().index[:3]:
        ev = run(P(intent="recommend", brand=brand), {})
        brands = list(ev.get("comparison", {}).get("specs", {}).get("brand", {}).values())
        check(f"recommend brand {brand}: every result is {brand}",
              brands and all(b == brand for b in brands))
    rng = random.Random(1)
    for pid in rng.sample(list(prod["id"]), min(3, len(prod))):
        base = prod[prod["id"] == pid].iloc[0]
        alts = T.find_alternatives(int(pid), max_price=1200, k=5)["alternatives"]
        ok = all(a["id"] != pid and a["price"] <= 1200 for a in alts)
        for a in alts:
            ok = ok and not T._same_config(base, prod[prod["id"] == a["id"]].iloc[0])
        check(f"alternatives for #{pid}: within budget, not itself, no duplicates", ok)
    check("compare ignores a laptop listed twice", T.compare([1, 1, 2])["product_ids"] == [1, 2])
    rev = T.search_reviews("battery", product_id=int(prod["id"].iloc[0]), k=5, min_score=0.0)
    check("review search stays inside the chosen laptop",
          all(m["product_id"] == int(prod["id"].iloc[0]) for m in rev["matches"]))
    check("unknown laptop gives a clear error, not a guess",
          "error" in run(P(intent="reviews", product_mentions=["Hoverboard 9000"]), {}))
    multi = prod["brand"].value_counts()
    brand = multi.index[0] if len(multi) and multi.iloc[0] > 1 else None
    if brand:
        check("ambiguous name asks the user to choose",
              "needs_choice" in run(P(intent="reviews", product_mentions=[brand]), {}))
    return checks


#  3. the planner
PLANNER_CASES = [
    ("Compare the Acer Nitro 5 and the Lenovo Legion 5", {"intent": "compare", "min_mentions": 2}),
    ("What do people say about battery life on the Razer Blade 15?",
     {"intent": "reviews", "aspect": "batter"}),
    ("Cheaper alternatives to the Acer Nitro 5 under $900",
     {"intent": "alternatives", "max_price": 900}),
    ("Best laptop under $1000 for gaming, I like Lenovo",
     {"intent": "recommend", "max_price": 1000, "brand": "lenovo"}),
    ("What is the weather in Delhi today?", {"intent": "off_topic"}),
    ("Which one is better?", {"intent": "clarify"}),
    ("Tell me a joke", {"intent": "off_topic"}),
    ("Is the keyboard good on the MSI GF63 Thin?", {"intent": "reviews", "aspect": "keyboard"}),
    ("Suggest something similar to the HP Pavilion 15.6", {"intent": "alternatives"}),
    ("Show me the best Asus laptop under $1500", {"intent": "recommend", "max_price": 1500,
                                                 "brand": "asus"}),
    ("Compare the Razer Blade 15 with the Alienware m15 for fan noise",
     {"intent": "compare", "min_mentions": 2}),
    ("How do I bake a cake?", {"intent": "off_topic"}),
]


def eval_planner():
    from src.agent.orchestrator import make_plan
    rows = []
    for q, exp in PLANNER_CASES:
        t0 = time.time()
        try:
            plan = make_plan(q, {})
        except Exception as e:  # quota or network problem: count it, don't crash
            rows.append((q, "ERROR", False, False, time.time() - t0, str(e)[:60]))
            continue
        intent_ok = plan.intent == exp["intent"]
        ok = intent_ok
        if "max_price" in exp:
            ok = ok and plan.max_price == exp["max_price"]
        if "brand" in exp:
            ok = ok and (plan.brand or "").lower() == exp["brand"]
        if "aspect" in exp:
            ok = ok and any(exp["aspect"] in a.lower() for a in plan.aspects)
        if "min_mentions" in exp:
            ok = ok and len(plan.product_mentions) >= exp["min_mentions"]
        rows.append((q, plan.intent, intent_ok, ok, time.time() - t0, ""))
        time.sleep(2)
    return rows


#  4. full answers (LLM)
E2E_QUESTIONS = [
    "Best laptop under $1000 for gaming",
    "What do reviewers say about battery life on the Razer Blade 15?",
    "Cheaper alternatives to the Razer Blade 15 under $1500",
    "Compare the Razer Blade 15 and the Acer Nitro 5",
    "Best laptop under $800 for a student",
    "What do reviewers say about the keyboard on the MSI GF63 Thin?",
]


def eval_answers(n):
    from src.agent.pipeline import handle
    rows = []
    for q in E2E_QUESTIONS[:n]:
        memory = {}
        t0 = time.time()
        r = handle(q, memory)
        if r["type"] == "choice":  # simulate the user picking the first option
            chosen = {m: opts[0]["id"] for m, opts in r["options"].items()}
            r = handle(q, memory, chosen=chosen, plan=r["plan"])
        secs = time.time() - t0
        if r["type"] != "answer":
            rows.append((q, r["type"], None, 0, secs, r["text"][:60]))
        else:
            cites = len(re.findall(r"\[R\d+\]", r["text"]))
            rows.append((q, "answer", not r["issues"], cites, secs, "; ".join(r["issues"])))
        time.sleep(4)
    return rows


#  5. spot-check sheet
def write_spot_check(path, n=15):
    prod = T._data()[0]
    sample = prod.sample(min(n, len(prod)), random_state=7)
    cols = ["id", "name", "ram_gb", "storage_gb", "storage_type", "cpu", "gpu", "screen_in"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols + ["all_correct (yes/no)"])
        for _, r in sample.iterrows():
            w.writerow([r[c] if str(r[c]) != "nan" else "" for c in cols] + [""])



def main():
    out = ["# Evaluation results", ""]

    rows, precision, hit_rate = eval_retrieval()
    out += ["## 1. Review retrieval (semantic search)", "",
            f"- Precision@5: **{precision:.0%}** of retrieved reviews mention the aspect",
            f"- Aspect hit rate: **{hit_rate:.0%}** of aspects returned at least one relevant review", "",
            "| Aspect | Relevant in top 5 |", "|---|---|"]
    out += [f"| {a} | {h}/{n} |" for a, h, n in rows] + [""]

    checks = eval_constraints()
    passed = sum(ok for _, ok in checks)
    out += ["## 2. Tool correctness (rules the tools must obey)", "",
            f"- **{passed}/{len(checks)}** checks passed", "", "| Check | Result |", "|---|---|"]
    out += [f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in checks] + [""]

    if RUN_LLM_PARTS:
        prow = eval_planner()
        done = [r for r in prow if r[1] != "ERROR"]
        out += ["## 3. Planner (understands the question)", "",
                f"- Intent accuracy: **{sum(r[2] for r in done)}/{len(done)}**",
                f"- Fully correct plans (intent, budget, brand, aspects): **{sum(r[3] for r in done)}/{len(done)}**",
                f"- Average planner latency: **{sum(r[4] for r in done) / max(1, len(done)):.1f}s**"]
        if len(done) < len(prow):
            out.append(f"- {len(prow) - len(done)} question(s) skipped because Gemini returned an error")
        out += ["", "| Question | Intent found | Correct |", "|---|---|---|"]
        out += [f"| {q} | {i} | {'yes' if ok else 'NO'} |" for q, i, _, ok, _, _ in prow] + [""]

        arow = eval_answers(ANSWER_QUESTIONS)
        ans = [r for r in arow if r[1] == "answer"]
        out += ["## 4. Full answers (grounding and speed)", "",
                f"- Answers produced: **{len(ans)}/{len(arow)}**",
                f"- Grounded (every citation and price matches the retrieved data): "
                f"**{sum(1 for r in ans if r[2])}/{len(ans)}**",
                f"- Average review citations per answer: **{sum(r[3] for r in ans) / max(1, len(ans)):.1f}**",
                f"- Average end-to-end time: **{sum(r[4] for r in arow) / max(1, len(arow)):.1f}s**",
                "", "| Question | Result | Grounded | Citations | Seconds |", "|---|---|---|---|---|"]
        out += [f"| {q} | {t} | {'yes' if g else ('n/a' if g is None else 'NO: ' + note)} | {c} | {s:.0f} |"
                for q, t, g, c, s, note in arow] + [""]

    write_spot_check(ROOT / "eval" / "spot_check.csv")
    out += ["## 5. Spec extraction spot-check", "",
            "Open `eval/spot_check.csv`, compare each row with the laptop's product title, and "
            "type yes or no in the last column. Accuracy = yes / total.", ""]
    (ROOT / "eval" / "results.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print("Saved eval/results.md and eval/spot_check.csv")


if __name__ == "__main__":
    main()