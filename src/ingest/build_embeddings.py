import sqlite3
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

con = sqlite3.connect("data/shop.db")
rev = pd.read_sql("SELECT id, product_id, title, text FROM reviews ORDER BY id", con)
prod = pd.read_sql("SELECT * FROM products ORDER BY id", con)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Reviews
rev_text = (rev["title"].fillna("") + ". " + rev["text"]).str.slice(0, 1000).tolist()
rev_emb = model.encode(rev_text, batch_size=64, normalize_embeddings=True,
                       show_progress_bar=True).astype("float32")

# Products (name + key specs + description)
def product_text(r):
    parts = [r["name"]]
    if pd.notna(r["ram_gb"]):
        parts.append(f"{int(r['ram_gb'])}GB RAM")
    if pd.notna(r["storage_gb"]):
        parts.append(f"{int(r['storage_gb'])}GB {r['storage_type'] or 'storage'}")
    if pd.notna(r["cpu"]):
        parts.append(f"CPU {r['cpu']}")
    if pd.notna(r["gpu"]):
        parts.append(f"GPU {r['gpu']}")
    if pd.notna(r["description"]):
        parts.append(r["description"])
    return ". ".join(parts)[:1500]
import html
import re

import pandas as pd
import streamlit as st

from src.agent.pipeline import handle

st.set_page_config(page_title="Laptop Buddy", page_icon="🧸", layout="wide")

# ---------------------------------------------------------------------------
# Currency: the data is in USD, the app shows rupees.
# Update USD_TO_INR whenever you want a fresher rate.
# ---------------------------------------------------------------------------
USD_TO_INR = 95


def inr(usd):
    """Format a dollar amount as rupees, e.g. 999 -> ₹94,905."""
    try:
        return f"₹{float(usd) * USD_TO_INR:,.0f}"
    except (TypeError, ValueError):
        return str(usd)


def usd_text_to_inr(text):
    """Turn '$1,299' inside the assistant's reply into rupee amounts."""
    return re.sub(r"\$\s?(\d[\d,]*(?:\.\d+)?)",
                  lambda m: inr(m.group(1).replace(",", "")), text)


def inr_text_to_usd(text):
    """Turn '₹80,000' typed by the user into dollars so the backend understands it."""
    return re.sub(r"(?:₹|Rs\.?\s?|INR\s?)\s?(\d[\d,]*(?:\.\d+)?)",
                  lambda m: f"${float(m.group(1).replace(',', '')) / USD_TO_INR:,.0f}",
                  text)


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: 'Nunito', sans-serif !important; }
.stApp { background: linear-gradient(160deg, #FFF5F9 0%, #F4F0FF 45%, #EAF6FF 100%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #FDE8F3 0%, #EDE7FF 100%); }
[data-testid="stHeader"] { background: transparent; }
.hero { text-align: center; padding: 8px 0 14px 0; }
.hero h1 { font-size: 2.4rem; font-weight: 800; color: #6B5B95; margin: 0; }
.hero p { color: #8A7FA3; font-size: 1.05rem; margin-top: 4px; }
[data-testid="stChatMessage"] { background: rgba(255,255,255,0.88); border-radius: 20px; padding: 14px 18px; border: 1px solid #F0E8FF; box-shadow: 0 4px 16px rgba(160,140,210,0.12); }
.stButton > button { width: 100%; background: #FFFFFF; color: #6B5B95; border: 1.5px solid #E3D8FF; border-radius: 999px; padding: 6px 16px; font-weight: 600; }
.stButton > button:hover { background: #F0E8FF; border-color: #C9B6FF; color: #4F3F7A; }
.chip { display: inline-block; background: #EDE7FF; color: #5B4B8A; border-radius: 999px; padding: 3px 12px; margin: 2px 4px 2px 0; font-size: .85rem; font-weight: 600; }
.chip.pink { background: #FFE3EE; color: #9B4A6B; }
.chip.mint { background: #DDF6EA; color: #3F7A5C; }
.chip.sky { background: #DDEEFF; color: #3F6A9B; }
.card { background: #fff; border-radius: 20px; padding: 16px; border: 1px solid #F0E8FF; box-shadow: 0 6px 18px rgba(160,140,210,.15); height: 100%; }
.card h4 { margin: 0 0 6px 0; color: #4A4458; font-size: .95rem; line-height: 1.3; }
.card .price { font-size: 1.4rem; font-weight: 800; color: #6B5B95; }
.card .stars { color: #F5B041; font-weight: 700; }
.card .note { color: #8A7FA3; font-size: .85rem; margin-top: 8px; }
.how { background: rgba(255,255,255,.7); border-radius: 16px; padding: 12px 14px; color: #6B5B95; font-size: .9rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

WELCOME = ("Hi there! 👋 I'm here to help you. Tell me your budget (in ₹) and what you'll use "
           "your laptop for, and I'll help you find a good match, using real specs and "
           "what other buyers actually say.")
EXAMPLES = [
    "Best laptop under ₹90000 for gaming",
    "Compare the Razer Blade 15 and the Acer Nitro 5",
    "What do reviewers say about battery life on the Razer Blade 15?",
    "Cheaper alternatives to the Razer Blade 15 under ₹140000",
]

ss = st.session_state
ss.setdefault("messages", [{"role": "assistant", "result": {"type": "text", "text": WELCOME}}])
ss.setdefault("memory", {})
ss.setdefault("pending", None)


def collect_reviews(obj, found=None):
    found = {} if found is None else found
    if isinstance(obj, dict):
        if "review_id" in obj and "text" in obj:
            found[obj["review_id"]] = obj
        for v in obj.values():
            collect_reviews(v, found)
    elif isinstance(obj, list):
        for v in obj:
            collect_reviews(v, found)
    return found


def fmt(v):
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def specs_table(comparison):
    specs = comparison["specs"]
    names = {i: f"#{i} {n[:40]}" for i, n in specs["name"].items()}
    df = pd.DataFrame({k: v for k, v in specs.items() if k != "name"}).T
    df = df.astype(object)
    df.columns = [names[c] for c in df.columns]
    if "price" in df.index:
        df.loc["price"] = df.loc["price"].map(inr)
    return df.apply(lambda col: col.map(fmt))


def chip(text, color=""):
    return f'<span class="chip {color}">{html.escape(str(text))}</span>'


def card(title, price, rating, chips, note=""):
    stars = "" if rating in (None, "not available") else f'<span class="stars">★ {rating}</span>'
    return (f'<div class="card"><h4>{html.escape(title)}</h4>'
            f'<div><span class="price">{inr(price)}</span> &nbsp; {stars}</div>'
            f'<div style="margin-top:8px">{"".join(chips)}</div>'
            f'<div class="note">{html.escape(note)}</div></div>')


def product_cards(ev):
    cards = []
    if ev.get("alternatives"):
        for a in ev["alternatives"]:
            cards.append(card(a["name"][:70], a["price"], a["rating"],
                              [chip(w, "mint") for w in a["why"][:3]],
                              f"{int(a['similarity'] * 100)}% similar"))
    elif "comparison" in ev:
        sp = ev["comparison"]["specs"]
        for pid, name in sp["name"].items():
            chips = []
            for key, label, color in [("ram_gb", "GB RAM", ""), ("storage_gb", "GB", "pink"),
                                      ("gpu", "", "sky"), ("cpu", "", "mint")]:
                v = sp[key][pid]
                if v != "not available":
                    chips.append(chip(f"{fmt(v)} {label}".strip(), color))
            cards.append(card(name[:70], sp["price"][pid], sp["rating"][pid], chips))
    if cards:
        cols = st.columns(min(len(cards), 4))
        for col, c in zip(cols, cards[:4]):
            col.markdown(c, unsafe_allow_html=True)


def render(r):
    st.markdown(usd_text_to_inr(r["text"]))
    if r["type"] != "answer":
        return
    ev = r["evidence"]
    product_cards(ev)
    if "comparison" in ev:
        with st.expander("📋 Every spec, side by side"):
            st.dataframe(specs_table(ev["comparison"]))
    reviews = collect_reviews(ev)
    if reviews:
        with st.expander(f"💬 What reviewers actually said ({len(reviews)})"):
            for rid, rv in sorted(reviews.items()):
                st.markdown(f"**[R{rid}]** {int(rv['stars'] or 0)}★ — {rv['text']}")
    if r["issues"]:
        st.caption("⚠️ Heads up, I couldn't fully verify: " + "; ".join(r["issues"]))
    else:
        st.caption("✓ I double-checked these prices and reviews against real data")
    with st.expander("🔍 Peek behind the scenes"):
        st.json(r["plan"].model_dump())


def respond(question, chosen=None, plan=None):
    with st.spinner("Flipping through the reviews for you..."):
        r = handle(inr_text_to_usd(question), ss.memory, chosen=chosen, plan=plan)
    if r["type"] == "choice":
        ss.pending = {"question": question, "plan": r["plan"], "options": r["options"]}
    else:
        ss.pending = None
        ss.messages.append({"role": "assistant", "result": r})


with st.sidebar:
    prefs_box = st.container()

st.markdown('<div class="hero"><h1>🧸 Laptop Buddy</h1>'
            '<p>Your friendly guide to finding the right laptop</p></div>',
            unsafe_allow_html=True)

if not any(m["role"] == "user" for m in ss.messages):
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex{i}"):
            ss.queued = ex

question = st.chat_input("Tell me what you're looking for...")
question = question or ss.pop("queued", None)
if question:
    ss.messages.append({"role": "user", "text": question})
    respond(question)

with prefs_box:
    st.markdown("### 🌸 What I know about you")
    mem = ss.memory
    chips = []
    if mem.get("budget"):
        chips.append(chip(f"💰 up to {inr(mem['budget'])}", "pink"))
    if mem.get("brand"):
        chips.append(chip(f"🏷️ {mem['brand']}", "sky"))
    if mem.get("use_case"):
        chips.append(chip(f"🎯 {mem['use_case']}", "mint"))
    if chips:
        st.markdown("".join(chips), unsafe_allow_html=True)
    else:
        st.caption("Nothing yet. Tell me a budget, brand or what you'll use it for!")
    st.caption(f"Prices shown in ₹ (converted at ₹{USD_TO_INR}/$)")
    if st.button("🧹 Start over"):
        ss.memory = {}
        ss.pending = None
        ss.messages = [{"role": "assistant", "result": {"type": "text", "text": WELCOME}}]
        st.rerun()

for m in ss.messages:
    with st.chat_message(m["role"], avatar="🌷" if m["role"] == "user" else "🧸"):
        if m["role"] == "user":
            st.markdown(m["text"])
        else:
            render(m["result"])

if ss.pending:
    p = ss.pending
    with st.chat_message("assistant", avatar="🧸"):
        st.markdown("I found a few that match. Which one did you mean? 🤔")
        picks = {}
        for mention, opts in p["options"].items():
            labels = {o["id"]: o["label"] for o in opts}
            picks[mention] = st.radio(f'"{mention}"', list(labels), key=f"pick_{mention}",
                                      format_func=lambda i, d=labels: d[i])
        if st.button("That one! ✨"):
            respond(p["question"], chosen=picks, plan=p["plan"])
            st.rerun()import html
import re

import pandas as pd
import streamlit as st

from src.agent.pipeline import handle

st.set_page_config(page_title="Laptop Buddy", page_icon="🧸", layout="wide")

# ---------------------------------------------------------------------------
# Currency: the data is in USD, the app shows rupees.
# Update USD_TO_INR whenever you want a fresher rate.
# ---------------------------------------------------------------------------
USD_TO_INR = 95


def inr(usd):
    """Format a dollar amount as rupees, e.g. 999 -> ₹94,905."""
    try:
        return f"₹{float(usd) * USD_TO_INR:,.0f}"
    except (TypeError, ValueError):
        return str(usd)


def usd_text_to_inr(text):
    """Turn '$1,299' inside the assistant's reply into rupee amounts."""
    return re.sub(r"\$\s?(\d[\d,]*(?:\.\d+)?)",
                  lambda m: inr(m.group(1).replace(",", "")), text)


def inr_text_to_usd(text):
    """Turn '₹80,000' typed by the user into dollars so the backend understands it."""
    return re.sub(r"(?:₹|Rs\.?\s?|INR\s?)\s?(\d[\d,]*(?:\.\d+)?)",
                  lambda m: f"${float(m.group(1).replace(',', '')) / USD_TO_INR:,.0f}",
                  text)


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: 'Nunito', sans-serif !important; }
.stApp { background: linear-gradient(160deg, #FFF5F9 0%, #F4F0FF 45%, #EAF6FF 100%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #FDE8F3 0%, #EDE7FF 100%); }
[data-testid="stHeader"] { background: transparent; }
.hero { text-align: center; padding: 8px 0 14px 0; }
.hero h1 { font-size: 2.4rem; font-weight: 800; color: #6B5B95; margin: 0; }
.hero p { color: #8A7FA3; font-size: 1.05rem; margin-top: 4px; }
[data-testid="stChatMessage"] { background: rgba(255,255,255,0.88); border-radius: 20px; padding: 14px 18px; border: 1px solid #F0E8FF; box-shadow: 0 4px 16px rgba(160,140,210,0.12); }
.stButton > button { width: 100%; background: #FFFFFF; color: #6B5B95; border: 1.5px solid #E3D8FF; border-radius: 999px; padding: 6px 16px; font-weight: 600; }
.stButton > button:hover { background: #F0E8FF; border-color: #C9B6FF; color: #4F3F7A; }
.chip { display: inline-block; background: #EDE7FF; color: #5B4B8A; border-radius: 999px; padding: 3px 12px; margin: 2px 4px 2px 0; font-size: .85rem; font-weight: 600; }
.chip.pink { background: #FFE3EE; color: #9B4A6B; }
.chip.mint { background: #DDF6EA; color: #3F7A5C; }
.chip.sky { background: #DDEEFF; color: #3F6A9B; }
.card { background: #fff; border-radius: 20px; padding: 16px; border: 1px solid #F0E8FF; box-shadow: 0 6px 18px rgba(160,140,210,.15); height: 100%; }
.card h4 { margin: 0 0 6px 0; color: #4A4458; font-size: .95rem; line-height: 1.3; }
.card .price { font-size: 1.4rem; font-weight: 800; color: #6B5B95; }
.card .stars { color: #F5B041; font-weight: 700; }
.card .note { color: #8A7FA3; font-size: .85rem; margin-top: 8px; }
.how { background: rgba(255,255,255,.7); border-radius: 16px; padding: 12px 14px; color: #6B5B95; font-size: .9rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

WELCOME = ("Hi there! 👋 I'm here to help you. Tell me your budget (in ₹) and what you'll use "
           "your laptop for, and I'll help you find a good match, using real specs and "
           "what other buyers actually say.")
EXAMPLES = [
    "Best laptop under ₹90000 for gaming",
    "Compare the Razer Blade 15 and the Acer Nitro 5",
    "What do reviewers say about battery life on the Razer Blade 15?",
    "Cheaper alternatives to the Razer Blade 15 under ₹140000",
]

ss = st.session_state
ss.setdefault("messages", [{"role": "assistant", "result": {"type": "text", "text": WELCOME}}])
ss.setdefault("memory", {})
ss.setdefault("pending", None)


def collect_reviews(obj, found=None):
    found = {} if found is None else found
    if isinstance(obj, dict):
        if "review_id" in obj and "text" in obj:
            found[obj["review_id"]] = obj
        for v in obj.values():
            collect_reviews(v, found)
    elif isinstance(obj, list):
        for v in obj:
            collect_reviews(v, found)
    return found


def fmt(v):
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def specs_table(comparison):
    specs = comparison["specs"]
    names = {i: f"#{i} {n[:40]}" for i, n in specs["name"].items()}
    df = pd.DataFrame({k: v for k, v in specs.items() if k != "name"}).T
    df = df.astype(object)
    df.columns = [names[c] for c in df.columns]
    if "price" in df.index:
        df.loc["price"] = df.loc["price"].map(inr)
    return df.apply(lambda col: col.map(fmt))


def chip(text, color=""):
    return f'<span class="chip {color}">{html.escape(str(text))}</span>'


def card(title, price, rating, chips, note=""):
    stars = "" if rating in (None, "not available") else f'<span class="stars">★ {rating}</span>'
    return (f'<div class="card"><h4>{html.escape(title)}</h4>'
            f'<div><span class="price">{inr(price)}</span> &nbsp; {stars}</div>'
            f'<div style="margin-top:8px">{"".join(chips)}</div>'
            f'<div class="note">{html.escape(note)}</div></div>')


def product_cards(ev):
    cards = []
    if ev.get("alternatives"):
        for a in ev["alternatives"]:
            cards.append(card(a["name"][:70], a["price"], a["rating"],
                              [chip(w, "mint") for w in a["why"][:3]],
                              f"{int(a['similarity'] * 100)}% similar"))
    elif "comparison" in ev:
        sp = ev["comparison"]["specs"]
        for pid, name in sp["name"].items():
            chips = []
            for key, label, color in [("ram_gb", "GB RAM", ""), ("storage_gb", "GB", "pink"),
                                      ("gpu", "", "sky"), ("cpu", "", "mint")]:
                v = sp[key][pid]
                if v != "not available":
                    chips.append(chip(f"{fmt(v)} {label}".strip(), color))
            cards.append(card(name[:70], sp["price"][pid], sp["rating"][pid], chips))
    if cards:
        cols = st.columns(min(len(cards), 4))
        for col, c in zip(cols, cards[:4]):
            col.markdown(c, unsafe_allow_html=True)


def render(r):
    st.markdown(usd_text_to_inr(r["text"]))
    if r["type"] != "answer":
        return
    ev = r["evidence"]
    product_cards(ev)
    if "comparison" in ev:
        with st.expander("📋 Every spec, side by side"):
            st.dataframe(specs_table(ev["comparison"]))
    reviews = collect_reviews(ev)
    if reviews:
        with st.expander(f"💬 What reviewers actually said ({len(reviews)})"):
            for rid, rv in sorted(reviews.items()):
                st.markdown(f"**[R{rid}]** {int(rv['stars'] or 0)}★ — {rv['text']}")
    if r["issues"]:
        st.caption("⚠️ Heads up, I couldn't fully verify: " + "; ".join(r["issues"]))
    else:
        st.caption("✓ I double-checked these prices and reviews against real data")
    with st.expander("🔍 Peek behind the scenes"):
        st.json(r["plan"].model_dump())


def respond(question, chosen=None, plan=None):
    with st.spinner("Flipping through the reviews for you..."):
        r = handle(inr_text_to_usd(question), ss.memory, chosen=chosen, plan=plan)
    if r["type"] == "choice":
        ss.pending = {"question": question, "plan": r["plan"], "options": r["options"]}
    else:
        ss.pending = None
        ss.messages.append({"role": "assistant", "result": r})


with st.sidebar:
    prefs_box = st.container()

st.markdown('<div class="hero"><h1>🧸 Laptop Buddy</h1>'
            '<p>Your friendly guide to finding the right laptop</p></div>',
            unsafe_allow_html=True)

if not any(m["role"] == "user" for m in ss.messages):
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex{i}"):
            ss.queued = ex

question = st.chat_input("Tell me what you're looking for...")
question = question or ss.pop("queued", None)
if question:
    ss.messages.append({"role": "user", "text": question})
    respond(question)

with prefs_box:
    st.markdown("### 🌸 What I know about you")
    mem = ss.memory
    chips = []
    if mem.get("budget"):
        chips.append(chip(f"💰 up to {inr(mem['budget'])}", "pink"))
    if mem.get("brand"):
        chips.append(chip(f"🏷️ {mem['brand']}", "sky"))
    if mem.get("use_case"):
        chips.append(chip(f"🎯 {mem['use_case']}", "mint"))
    if chips:
        st.markdown("".join(chips), unsafe_allow_html=True)
    else:
        st.caption("Nothing yet. Tell me a budget, brand or what you'll use it for!")
    st.caption(f"Prices shown in ₹ (converted at ₹{USD_TO_INR}/$)")
    if st.button("🧹 Start over"):
        ss.memory = {}
        ss.pending = None
        ss.messages = [{"role": "assistant", "result": {"type": "text", "text": WELCOME}}]
        st.rerun()

for m in ss.messages:
    with st.chat_message(m["role"], avatar="🌷" if m["role"] == "user" else "🧸"):
        if m["role"] == "user":
            st.markdown(m["text"])
        else:
            render(m["result"])

if ss.pending:
    p = ss.pending
    with st.chat_message("assistant", avatar="🧸"):
        st.markdown("I found a few that match. Which one did you mean? 🤔")
        picks = {}
        for mention, opts in p["options"].items():
            labels = {o["id"]: o["label"] for o in opts}
            picks[mention] = st.radio(f'"{mention}"', list(labels), key=f"pick_{mention}",
                                      format_func=lambda i, d=labels: d[i])
        if st.button("That one! ✨"):
            respond(p["question"], chosen=picks, plan=p["plan"])
            st.rerun()
prod_emb = model.encode([product_text(r) for _, r in prod.iterrows()],
                        batch_size=32, normalize_embeddings=True).astype("float32")

np.save("data/review_emb.npy", rev_emb)
np.save("data/review_ids.npy", rev["id"].values)
np.save("data/product_emb.npy", prod_emb)
np.save("data/product_ids.npy", prod["id"].values)
print("saved:", rev_emb.shape, prod_emb.shape)

# Quick semantic search test (cosine similarity = dot product on normalized vectors)
for q in ["battery drains fast", "gets very hot and loud fans", "great screen for the price"]:
    qv = model.encode([q], normalize_embeddings=True)[0]
    scores = rev_emb @ qv
    print("\nQUERY:", q)
    for i in np.argsort(-scores)[:4]:
        print(f"  {scores[i]:.2f} | {rev.iloc[i]['text'][:140]}")