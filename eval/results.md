# Evaluation results

## 1. Review retrieval (semantic search)

- Precision@5: **78%** of retrieved reviews mention the aspect
- Aspect hit rate: **100%** of aspects returned at least one relevant review

| Aspect | Relevant in top 5 |
|---|---|
| battery life | 3/5 |
| fan noise and heat | 5/5 |
| screen quality | 4/5 |
| keyboard | 5/5 |
| gaming performance | 5/5 |
| build quality | 4/5 |
| speakers and sound | 4/5 |
| value for money | 1/5 |

## 2. Tool correctness (rules the tools must obey)

- **13/13** checks passed

| Check | Result |
|---|---|
| recommend under $800: every result within budget | PASS |
| recommend under $1000: every result within budget | PASS |
| recommend under $1500: every result within budget | PASS |
| recommend brand ASUS: every result is ASUS | PASS |
| recommend brand Acer: every result is Acer | PASS |
| recommend brand Lenovo: every result is Lenovo | PASS |
| alternatives for #18: within budget, not itself, no duplicates | PASS |
| alternatives for #73: within budget, not itself, no duplicates | PASS |
| alternatives for #9: within budget, not itself, no duplicates | PASS |
| compare ignores a laptop listed twice | PASS |
| review search stays inside the chosen laptop | PASS |
| unknown laptop gives a clear error, not a guess | PASS |
| ambiguous name asks the user to choose | PASS |

## 3. Planner (understands the question)

- Intent accuracy: **12/12**
- Fully correct plans (intent, budget, brand, aspects): **12/12**
- Average planner latency: **1.3s**

| Question | Intent found | Correct |
|---|---|---|
| Compare the Acer Nitro 5 and the Lenovo Legion 5 | compare | yes |
| What do people say about battery life on the Razer Blade 15? | reviews | yes |
| Cheaper alternatives to the Acer Nitro 5 under $900 | alternatives | yes |
| Best laptop under $1000 for gaming, I like Lenovo | recommend | yes |
| What is the weather in Delhi today? | off_topic | yes |
| Which one is better? | clarify | yes |
| Tell me a joke | off_topic | yes |
| Is the keyboard good on the MSI GF63 Thin? | reviews | yes |
| Suggest something similar to the HP Pavilion 15.6 | alternatives | yes |
| Show me the best Asus laptop under $1500 | recommend | yes |
| Compare the Razer Blade 15 with the Alienware m15 for fan noise | compare | yes |
| How do I bake a cake? | off_topic | yes |

## 4. Full answers (grounding and speed)

- Answers produced: **3/3**
- Grounded (every citation and price matches the retrieved data): **3/3**
- Average review citations per answer: **3.0**
- Average end-to-end time: **10.2s**

| Question | Result | Grounded | Citations | Seconds |
|---|---|---|---|---|
| Best laptop under $1000 for gaming | answer | yes | 7 | 11 |
| What do reviewers say about battery life on the Razer Blade 15? | answer | yes | 2 | 7 |
| Cheaper alternatives to the Razer Blade 15 under $1500 | answer | yes | 0 | 13 |

## 5. Spec extraction spot-check

Open `eval/spot_check.csv`, compare each row with the laptop's product title, and type yes or no in the last column. Accuracy = yes / total.
