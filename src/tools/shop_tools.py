import re
import sqlite3
from functools import lru_cache

import numpy as np
import pandas as pd

DB_PATH = "data/shop.db"
NA = "not available"


@lru_cache(maxsize=1)
def _model():
    try:  # light ONNX runtime, no torch needed (used when deployed)
        from fastembed import TextEmbedding
        return ("fast", TextEmbedding("sentence-transformers/all-MiniLM-L6-v2"))
    except ImportError:
        from sentence_transformers import SentenceTransformer
        return ("st", SentenceTransformer("all-MiniLM-L6-v2"))


def embed(text):
    kind, m = _model()
    if kind == "fast":
        v = np.asarray(next(iter(m.embed([text]))), dtype="float32")
        return v / np.linalg.norm(v)
    return m.encode([text], normalize_embeddings=True)[0]

@lru_cache(maxsize=1)
def _data():
    con = sqlite3.connect(DB_PATH)
    prod = pd.read_sql("SELECT * FROM products ORDER BY id", con)
    rev = pd.read_sql("SELECT * FROM reviews ORDER BY id", con)
    con.close()
    return prod, rev, np.load("data/review_emb.npy"), np.load("data/product_emb.npy")


def _ok(x):
    return x is not None and not pd.isna(x)


def _py(x):
    """numpy -> plain python, NaN -> None"""
    if not _ok(x):
        return None
    return x.item() if hasattr(x, "item") else x


# ---------- helper: find a product by id or name words ----------
def lookup(query):
    prod = _data()[0]
    if isinstance(query, int) or str(query).isdigit():
        return prod[prod["id"] == int(query)]
    mask = np.ones(len(prod), dtype=bool)
    for w in str(query).lower().split():
        mask &= prod["name"].str.lower().str.contains(re.escape(w), na=False).values
    return prod[mask]


# ---------- tool 1: compare ----------
SPEC_COLS = ["name", "brand", "price", "rating", "rating_count", "ram_gb",
             "storage_gb", "storage_type", "cpu", "gpu", "screen_in",
             "refresh_hz", "os"]


def compare(product_ids):
    prod = _data()[0].set_index("id")
    missing = [i for i in product_ids if i not in prod.index]
    ids = [i for i in dict.fromkeys(product_ids) if i in prod.index]
    sel = prod.loc[ids]
    table = {c: {int(i): (_py(sel.loc[i, c]) if _ok(sel.loc[i, c]) else NA) for i in ids}
             for c in SPEC_COLS}

    def best(col, fn):
        s = sel[col].dropna()
        return int(fn(s)) if len(s) else None

    highlights = {
        "cheapest": best("price", lambda s: s.idxmin()),
        "best_rated": best("rating", lambda s: s.idxmax()),
        "most_ram": best("ram_gb", lambda s: s.idxmax()),
        "most_storage": best("storage_gb", lambda s: s.idxmax()),
    }
    return {"product_ids": ids, "missing_ids": missing, "specs": table,
            "highlights": highlights}


# ---------- tool 2: review search (RAG retrieval) ----------
def search_reviews(aspect, product_id=None, k=5, min_score=0.30):
    prod, rev, rev_emb, _ = _data()
    scores = rev_emb @ embed(aspect)
    if product_id is not None:
        scores = np.where(rev["product_id"].values == product_id, scores, -1.0)
    out = []
    for i in np.argsort(-scores)[:k]:
        if scores[i] < min_score:
            break
        r = rev.iloc[i]
        out.append({"review_id": int(r["id"]), "product_id": int(r["product_id"]),
                    "score": round(float(scores[i]), 3), "stars": _py(r["rating"]),
                    "verified": bool(r["verified"]), "text": r["text"][:400]})
    avg = round(float(np.mean([o["stars"] for o in out])), 2) if out else None
    return {"aspect": aspect, "product_id": product_id, "matches": out,
            "avg_stars_of_matches": avg}


# ---------- tool 3: alternatives ----------
def _why(base, alt):
    why = []
    d = alt["price"] - base["price"]
    if abs(d) >= 1:
        why.append(f"${abs(d):,.0f} {'cheaper' if d < 0 else 'more expensive'}")
    if _ok(alt["ram_gb"]) and _ok(base["ram_gb"]) and alt["ram_gb"] != base["ram_gb"]:
        why.append(f"{int(alt['ram_gb'])}GB RAM vs {int(base['ram_gb'])}GB")
    if _ok(alt["gpu"]) and _ok(base["gpu"]) and alt["gpu"] != base["gpu"]:
        why.append(f"GPU {alt['gpu']} vs {base['gpu']}")
    if _ok(alt["rating"]) and _ok(base["rating"]) and abs(alt["rating"] - base["rating"]) >= 0.1:
        why.append(f"rated {alt['rating']:.1f} vs {base['rating']:.1f}")
    return why


def find_alternatives(product_id, max_price=None, min_rating=None,
                      same_brand_only=False, k=5):
    prod, _, _, prod_emb = _data()
    idx = prod.index[prod["id"] == product_id]
    if len(idx) == 0:
        return {"error": f"product {product_id} not found", "alternatives": []}
    base = prod.loc[idx[0]]
    sims = prod_emb @ prod_emb[idx[0]]
    out = []
    for i in np.argsort(-sims):
        a = prod.iloc[i]
        if a["id"] == product_id:
            continue
        if max_price is not None and a["price"] > max_price:
            continue
        if min_rating is not None and (not _ok(a["rating"]) or a["rating"] < min_rating):
            continue
        if same_brand_only and a["brand"] != base["brand"]:
            continue
        out.append({"id": int(a["id"]), "name": a["name"], "price": float(a["price"]),
                    "rating": _py(a["rating"]), "similarity": round(float(sims[i]), 3),
                    "why": _why(base, a)})
        if len(out) == k:
            break
    return {"base_id": product_id, "alternatives": out}


# ---------- manual test ----------
if __name__ == "__main__":
    import json
    print("lookup 'legion 5':")
    print(lookup("legion 5")[["id", "name"]].to_string())
    c = compare([1, 2, 3])
    print("\ncompare 1,2,3 highlights:", json.dumps(c["highlights"]))
    for col in ("price", "ram_gb", "gpu"):
        print(" ", col, c["specs"][col])
    print("\nreviews about battery for product 1:")
    for m in search_reviews("battery life", product_id=1, k=3)["matches"]:
        print(" ", m["score"], m["stars"], m["text"][:120])
    print("\nalternatives to product 1 under $1200:")
    for a in find_alternatives(1, max_price=1200, k=3)["alternatives"]:
        print(" ", a["id"], a["price"], a["similarity"], a["why"])