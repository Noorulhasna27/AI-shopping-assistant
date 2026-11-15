from src.agent.orchestrator import make_plan

QUESTIONS = [
    "Compare the Acer Nitro 5 and the Lenovo Legion 5",
    "What do people say about battery life on the Razer Blade 15?",
    "Cheaper alternatives to the Acer Nitro 5 under $900",
    "Best laptop under $1000 for gaming, I like Lenovo",
    "What is the weather in Delhi today?",
    "Which one is better?",
]

for q in QUESTIONS:
    print("\nQ:", q)
    print(make_plan(q, {}).model_dump_json())
