import hashlib
import html
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent.pipeline import handle

st.set_page_config(page_title="Laptop Shopping Assistant", page_icon="💻", layout="wide")

USD_TO_INR = 95

SHORTLIST_FILE = Path(__file__).parent / "shortlist.json"
PERSIST_SHORTLIST = False


def inr(usd):
    """Format a dollar amount as rupees with Indian digit grouping, e.g. 999 -> ₹94,905."""
    try:
        n = int(round(float(usd) * USD_TO_INR))
    except (TypeError, ValueError):
        return str(usd)
    digits = str(abs(n))
    if len(digits) > 3:
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", digits[:-3])
        digits = head + "," + digits[-3:]
    return ("-" if n < 0 else "") + "₹" + digits


_NUM = r"(\d+(?:,\d{2,3})*(?:\.\d+)?)(?:\s*(k|lakhs?|lac)\b)?"


def usd_text_to_inr(text):
    """Show everything in rupees: $ amounts become INR, and 'dollars'/'USD' wording becomes rupees."""
    text = re.sub(r"\$\s?(\d+(?:,\d{3})*(?:\.\d+)?)",
                  lambda m: inr(m.group(1).replace(",", "")), text)

    def word(m):
        w = m.group(1)
        return "Rupees" if w[:2].upper() != "US" and w[0].isupper() else "rupees"

    text = re.sub(r"\b(US\s+dollars?|dollars?)\b", word, text, flags=re.I)
    return re.sub(r"\bUSD\b", "INR", text)


def inr_text_to_usd(text):
    """Turn rupee amounts typed by the user into dollars so the backend understands them.

    Handles: ₹90000, Rs 90,000, INR 90000, 90000 rupees, 90k rupees, 1.5 lakh rupees,
    1,50,000, and a bare 90000 (any plain number of 10,000 or more counts as rupees).
    Amounts written with a $ sign are left alone.
    """
    def value_of(m):
        v = float(m.group(1).replace(",", ""))
        unit = (m.group(2) or "").lower()
        if unit == "k":
            v *= 1_000
        elif unit in ("lakh", "lakhs", "lac"):
            v *= 100_000
        return v

    def explicit(m):
        return f"${value_of(m) / USD_TO_INR:,.0f}"

    def bare(m):
        v = value_of(m)
        return f"${v / USD_TO_INR:,.0f}" if v >= 10_000 else m.group(0)

    before = re.compile(r"(?:₹|\brs\.?|\binr)\s*" + _NUM, re.I)
    after = re.compile(_NUM + r"\s*(?:rupees?|\brs\b\.?|\binr\b|₹)", re.I)
    units = r"(?!\s*(?:gb|tb|mb|mah|wh|hz|mhz|ghz|px|nits|mm|cm|inch)\b)"
    loose = re.compile(r"(?<![\w$.,₹-])" + _NUM + r"(?![A-Za-z\d])" + units, re.I)

    text = before.sub(explicit, text)
    text = after.sub(explicit, text)
    return loose.sub(bare, text)


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
  --ink: #1E2440;
  --muted: #5F6785;
  --line: #DFE3F4;
  --accent: #4B49C8;
  --accent-dark: #3837A8;
  --accent-soft: #ECEBFF;
}

.stApp { font-family: 'Plus Jakarta Sans', sans-serif; background: linear-gradient(180deg, #F5F3FF 0%, #EEF3FF 45%, #F6F9FC 100%); }
.stMarkdown, .stCaption, button, input, textarea, [data-testid="stChatInput"] { font-family: 'Plus Jakarta Sans', sans-serif !important; }
.stMarkdown, .stMarkdown p, .stMarkdown li { color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }

/* Sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #ECE9FB 0%, #E3ECFB 100%); border-right: 1px solid var(--line); }
.side-title { font-size: .72rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; color: var(--accent); margin: 4px 0 10px 0; }
.side-note { color: var(--muted); font-size: .8rem; margin: 10px 0 14px 0; line-height: 1.5; }
.short-item { background: #FFFFFF; border: 1px solid var(--line); border-radius: 10px; padding: 8px 12px; margin: 8px 0 4px 0; font-size: .82rem; line-height: 1.4; color: var(--ink); }
.short-item span { color: var(--accent); font-weight: 800; }
.short-item .short-note { display: block; margin-top: 4px; color: var(--muted); font-size: .78rem; font-weight: 500; font-style: italic; }

/* Hero */
.hero { text-align: center; padding: 6px 0 22px 0; }
.hero h1 { font-size: 2.7rem; font-weight: 800; color: var(--ink); letter-spacing: -0.03em; margin: 6px 0 8px 0; line-height: 1.1; }
.hero p { color: var(--muted); font-size: 1.05rem; max-width: 640px; margin: 0 auto; line-height: 1.55; }
.trust { display: flex; justify-content: center; gap: 22px; flex-wrap: wrap; margin-top: 16px; color: var(--muted); font-size: .82rem; font-weight: 600; }
.trust span::before { content: ""; display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--accent); margin-right: 8px; vertical-align: middle; }
.section-label { font-size: .72rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin: 6px 0 10px 0; }

/* Chat */
[data-testid="stChatMessage"] { background: #FFFFFF; border: 1px solid var(--line); border-radius: 14px; padding: 18px 22px; box-shadow: 0 1px 2px rgba(30,36,64,.04), 0 8px 24px rgba(75,73,200,.06); }
[data-testid="stChatInput"] { border-radius: 14px; }

/* Buttons */
.stButton > button { width: 100%; background: #FFFFFF; color: var(--ink); border: 1px solid var(--line); border-radius: 12px; padding: 10px 16px; font-weight: 600; text-align: left; transition: all .15s ease; }
.stButton > button:hover { background: var(--accent-soft); border-color: var(--accent); color: var(--accent-dark); }
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] { background: var(--accent); color: #FFFFFF; border: 1px solid var(--accent); font-weight: 700; text-align: center; }
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover { background: var(--accent-dark); border-color: var(--accent-dark); color: #FFFFFF; }

/* Expanders */
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: #FBFBFF; }

/* Chips */
.chip { display: inline-block; background: #EAE6FF; color: #3D3A8C; border-radius: 8px; padding: 4px 10px; margin: 2px 6px 2px 0; font-size: .78rem; font-weight: 600; }
.chip.pink { background: #FCE7EF; color: #8A3555; }
.chip.mint { background: #DDF3EA; color: #22654A; }
.chip.sky { background: #E0EDFF; color: #1E4E8C; }

/* Product cards */
.card { background: #FFFFFF; border: 1px solid var(--line); border-top: 4px solid var(--accent); border-radius: 14px; padding: 18px; height: 100%; box-shadow: 0 1px 2px rgba(30,36,64,.04), 0 10px 26px rgba(75,73,200,.08); }
.card .rank { font-size: .68rem; font-weight: 800; letter-spacing: .12em; color: var(--accent); margin-bottom: 8px; }
.card h4 { margin: 0 0 12px 0; color: var(--ink); font-size: .95rem; font-weight: 700; line-height: 1.35; }
.card .price { font-size: 1.5rem; font-weight: 800; color: var(--ink); letter-spacing: -0.02em; }
.card .stars { color: #B7791F; font-weight: 700; font-size: .9rem; }
.card .note { color: var(--muted); font-size: .82rem; margin-top: 12px; font-weight: 600; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

WELCOME = ("Welcome. Tell me your budget in ₹ and what you will use the laptop for. "
           "I will recommend options based on verified specs and real customer reviews. "
           "Use Save to shortlist on any result to keep it for later.")
EXAMPLES = [
    "Best laptop under ₹90000 for gaming",
    "Compare the Razer Blade 15 and the Acer Nitro 5",
    "What do reviewers say about battery life on the Razer Blade 15?",
    "Cheaper alternatives to the Razer Blade 15 under ₹140000",
]


def load_shortlist():
    if not PERSIST_SHORTLIST:
        return {}
    try:
        data = json.loads(SHORTLIST_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def persist_shortlist():
    if not PERSIST_SHORTLIST:
        return
    try:
        SHORTLIST_FILE.write_text(json.dumps(ss.shortlist, indent=2), encoding="utf-8")
    except Exception:
        pass 


ss = st.session_state
ss.setdefault("messages", [{"role": "assistant", "result": {"type": "text", "text": WELCOME}}])
ss.setdefault("memory", {})
ss.setdefault("pending", None)
if "shortlist" not in ss:
    ss.shortlist = load_shortlist()


def toggle_save(item):
    name = item["name"]
    if name in ss.shortlist:
        del ss.shortlist[name]
    else:
        ss.shortlist[name] = item
    persist_shortlist()


def remove_saved(name):
    ss.shortlist.pop(name, None)
    persist_shortlist()


def clear_shortlist():
    ss.shortlist = {}
    persist_shortlist()


def note_key(name):
    """Stable widget key for a laptop's note box (safe even if the list order changes)."""
    return "note_" + hashlib.md5(name.encode("utf-8")).hexdigest()[:10]


def save_note(name):
    if name in ss.shortlist:
        ss.shortlist[name]["note"] = ss.get(note_key(name), "").strip()
        persist_shortlist()


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


def card(title, price, rating, chips, note="", rank=None):
    stars = "" if rating in (None, "not available") else f'<span class="stars">★ {rating}</span>'
    rank_html = f'<div class="rank">OPTION {rank}</div>' if rank else ""
    return (f'<div class="card">{rank_html}<h4>{html.escape(title)}</h4>'
            f'<div><span class="price">{inr(price)}</span> &nbsp; {stars}</div>'
            f'<div style="margin-top:10px">{"".join(chips)}</div>'
            f'<div class="note">{html.escape(note)}</div></div>')


def product_cards(ev, idx):
    entries = []  # (card_html, item_to_save)
    if ev.get("alternatives"):
        for n, a in enumerate(ev["alternatives"], 1):
            why = [str(w) for w in a["why"][:3]]
            html_card = card(a["name"][:70], a["price"], a["rating"],
                             [chip(w, "mint") for w in why],
                             f"{int(a['similarity'] * 100)}% similarity match", rank=n)
            item = {"name": f"#{a['id']} {a['name'][:66]}", "price": a["price"],
                    "rating": a["rating"], "details": ", ".join(why)}
            entries.append((html_card, item))
    elif "comparison" in ev:
        sp = ev["comparison"]["specs"]
        for n, (pid, name) in enumerate(sp["name"].items(), 1):
            tags = []
            for key, label, color in [("ram_gb", "GB RAM", ""), ("storage_gb", "GB", "pink"),
                                      ("gpu", "", "sky"), ("cpu", "", "mint")]:
                v = sp[key][pid]
                if v != "not available":
                    tags.append((f"{fmt(v)} {label}".strip(), color))
            html_card = card(name[:70], sp["price"][pid], sp["rating"][pid],
                             [chip(t, c) for t, c in tags], rank=n)
            item = {"name": f"#{pid} {name[:66]}", "price": sp["price"][pid],
                    "rating": sp["rating"][pid], "details": ", ".join(t for t, _ in tags)}
            entries.append((html_card, item))
    if entries:
        cols = st.columns(min(len(entries), 4))
        for n, (col, (html_card, item)) in enumerate(zip(cols, entries[:4])):
            col.markdown(html_card, unsafe_allow_html=True)
            saved = item["name"] in ss.shortlist
            col.button("Remove from shortlist" if saved else "Save to shortlist",
                       key=f"save_{idx}_{n}", on_click=toggle_save, args=(item,))


def render(r, idx):
    st.markdown(usd_text_to_inr(r["text"]))
    if r["type"] != "answer":
        return
    ev = r["evidence"]
    product_cards(ev, idx)
    if "comparison" in ev:
        with st.expander("Full specification comparison"):
            st.dataframe(specs_table(ev["comparison"]))
    reviews = collect_reviews(ev)
    if reviews:
        with st.expander(f"Customer reviews ({len(reviews)})"):
            for rid, rv in sorted(reviews.items()):
                st.markdown(f"**[R{rid}]** {int(rv['stars'] or 0)}★ — {rv['text']}")
    if r["issues"]:
        st.caption("Could not fully verify: " + "; ".join(r["issues"]))
    else:
        st.caption("Prices and review quotes verified against source data.")
    with st.expander("How this answer was built"):
        st.json(r["plan"].model_dump())


def respond(question, chosen=None, plan=None):
    with st.spinner("Analyzing specs and customer reviews..."):
        r = handle(inr_text_to_usd(question), ss.memory, chosen=chosen, plan=plan)
    if r["type"] == "choice":
        ss.pending = {"question": question, "plan": r["plan"], "options": r["options"]}
    else:
        ss.pending = None
        ss.messages.append({"role": "assistant", "result": r})


# Page layout

with st.sidebar:
    prefs_box = st.container()
    short_box = st.container()

st.markdown(
    '<div class="hero">'
    '<h1>Laptop Shopping Assistant</h1>'
    '<p>Share your budget and what you need. Get ranked recommendations backed by '
    'verified specifications and real customer reviews.</p>'
    '<div class="trust"><span>Verified specs</span><span>Real customer reviews</span></div>'
    '</div>',
    unsafe_allow_html=True,
)

if not any(m["role"] == "user" for m in ss.messages):
    st.markdown('<div class="section-label">Try a sample question</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex{i}"):
            ss.queued = ex

question = st.chat_input("e.g. Best gaming laptop under ₹90000, or compare two models")
question = question or ss.pop("queued", None)
if question:
    ss.messages.append({"role": "user", "text": question})
    respond(question)

with prefs_box:
    st.markdown('<div class="side-title">Your preferences</div>', unsafe_allow_html=True)
    mem = ss.memory
    chips = []
    if mem.get("budget"):
        chips.append(chip(f"Budget: up to {inr(mem['budget'])}", "pink"))
    if mem.get("brand"):
        chips.append(chip(f"Brand: {mem['brand']}", "sky"))
    if mem.get("use_case"):
        chips.append(chip(f"Use case: {mem['use_case']}", "mint"))
    if chips:
        st.markdown("".join(chips), unsafe_allow_html=True)
    else:
        st.caption("No preferences captured yet. Mention a budget, brand, or use case.")
    if st.button("Reset conversation"):
        ss.memory = {}
        ss.pending = None
        ss.messages = [{"role": "assistant", "result": {"type": "text", "text": WELCOME}}]
        st.rerun()

with short_box:
    st.divider()
    st.markdown(f'<div class="side-title">Shortlist ({len(ss.shortlist)})</div>',
                unsafe_allow_html=True)
    if not ss.shortlist:
        st.caption("Nothing saved yet. Use Save to shortlist on any result to review it later.")
    else:
        for i, (name, it) in enumerate(ss.shortlist.items()):
            st.markdown(f'<div class="short-item">{html.escape(name)}<br>'
                        f'<span>{inr(it["price"])}</span></div>', unsafe_allow_html=True)
            st.text_input("Note", value=it.get("note", ""), key=note_key(name),
                          placeholder="Add a note, then press Enter",
                          label_visibility="collapsed",
                          on_change=save_note, args=(name,))
            st.button("Remove", key=f"rm_{i}", on_click=remove_saved, args=(name,))
        st.button("Clear shortlist", key="clear_short", on_click=clear_shortlist)

if ss.shortlist:
    with st.expander(f"Your shortlist ({len(ss.shortlist)}): review and compare", expanded=False):
        df = pd.DataFrame([
            {"Laptop": n, "Price": inr(it["price"]), "Rating": str(it["rating"]),
             "Key specs": it["details"], "Note": it.get("note", "")}
            for n, it in ss.shortlist.items()
        ])
        st.dataframe(df, hide_index=True)
        st.download_button("Download shortlist (CSV)", df.to_csv(index=False).encode("utf-8"),
                           file_name="shortlist.csv", mime="text/csv")


for idx, m in enumerate(ss.messages):
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["text"])
        else:
            render(m["result"], idx)

if ss.pending:
    p = ss.pending
    with st.chat_message("assistant"):
        st.markdown("I found several matches. Please confirm which product you mean.")
        picks = {}
        for mention, opts in p["options"].items():
            labels = {o["id"]: o["label"] for o in opts}
            picks[mention] = st.radio(f'"{mention}"', list(labels), key=f"pick_{mention}",
                                      format_func=lambda i, d=labels: d[i])
        if st.button("Confirm selection", type="primary"):
            respond(p["question"], chosen=picks, plan=p["plan"])
            st.rerun()