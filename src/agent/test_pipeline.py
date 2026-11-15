from src.agent.pipeline import handle

memory = {}
QUESTIONS = [
    "Best laptop under $1000 for gaming",
    "What do people say about battery life on the Razer Blade 15?",
    "Is the weather nice today?",
]

for q in QUESTIONS:
    print("\nQ:", q)
    r = handle(q, memory)
    if r["type"] == "choice":
        chosen = {m: opts[0]["id"] for m, opts in r["options"].items()}
        print("  (several matches, auto-picking the first):", chosen)
        r = handle(q, memory, chosen=chosen, plan=r["plan"])
    print(r["text"])
    print("  issues:", r.get("issues"), "| memory:", memory)
