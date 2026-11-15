from src.tools import shop_tools as T

DEFAULT_ASPECTS = ["battery life", "display", "performance", "build quality"]


def _short(name, n=80):
    return name if len(name) <= n else name[:n].rstrip() + "..."


def resolve(mentions, chosen=None):
    """Map product names to ids. Returns (ids, needs_choice, not_found)."""
    chosen = chosen or {}
    ids, needs_choice, not_found = [], {}, []
    for m in mentions:
        if m in chosen:
            ids.append(int(chosen[m]))
            continue
        hits = T.lookup(m)
        if len(hits) == 0:
            not_found.append(m)
        elif len(hits) == 1:
            ids.append(int(hits.iloc[0]["id"]))
        else:
            hits = hits.sort_values("rating_count", ascending=False).head(6)
            needs_choice[m] = [
                {"id": int(r["id"]), "label": f"{_short(r['name'])} (${r['price']:,.0f})"}
                for _, r in hits.iterrows()]
    return ids, needs_choice, not_found


def _snips(aspect, pid, k=2):
    return T.search_reviews(aspect, product_id=pid, k=k)["matches"]


def _breakdown(pid):
    rev = T._data()[1]
    r = rev[rev["product_id"] == pid]["rating"]
    return {"reviews_in_db": int(len(r)),
            "avg_stars": round(float(r.mean()), 2) if len(r) else None,
            "stars": {str(s): int((r.round() == s).sum()) for s in range(1, 6)}}


def _compare(plan, ids):
    aspects = plan.aspects[:2] or ([plan.use_case] if plan.use_case else [])
    return {"intent": "compare", "comparison": T.compare(ids),
            "products": {pid: {"reviews": _breakdown(pid),
                               "aspect_snippets": {a: _snips(a, pid) for a in aspects}}
                         for pid in ids}}


def _reviews(plan, pid):
    aspects = plan.aspects[:4] or DEFAULT_ASPECTS
    return {"intent": "reviews", "product": T.compare([pid]),
            "reviews": _breakdown(pid),
            "aspect_snippets": {a: _snips(a, pid, 3) for a in aspects}}


def _alternatives(pid, budget):
    alts = T.find_alternatives(pid, max_price=budget, k=4)["alternatives"]
    for a in alts:
        a["reviews"] = _breakdown(a["id"])
    return {"intent": "alternatives", "base": T.compare([pid]),
            "max_price": budget, "alternatives": alts}


def _recommend(plan, budget, brand):
    prod, rev = T._data()[0], T._data()[1]
    df = prod
    if budget:
        df = df[df["price"] <= budget]
    if brand:
        df = df[df["brand"].str.lower() == brand.lower()]
    if df.empty:
        return {"error": "No laptops in the catalog match that budget and brand. "
                         "Try a higher budget or another brand."}
    avg = rev.groupby("product_id")["rating"].mean()
    df = df.assign(avg_stars=df["id"].map(avg))
    df = df.sort_values(["avg_stars", "rating_count"], ascending=False).head(4)
    ids = [int(i) for i in df["id"]]
    use = plan.use_case or (plan.aspects[0] if plan.aspects else None)
    return {"intent": "recommend",
            "filters": {"max_price": budget, "brand": brand, "use_case": use},
            "ranking_rule": "average review stars in our review set, then Amazon rating count",
            "comparison": T.compare(ids),
            "products": {pid: {"reviews": _breakdown(pid),
                               "snippets": _snips(use, pid) if use else []}
                         for pid in ids}}


def run(plan, memory, chosen=None):
    budget = plan.max_price or memory.get("budget")
    brand = plan.brand or memory.get("brand")
    if plan.intent == "recommend":
        return _recommend(plan, budget, brand)
    ids, needs_choice, not_found = resolve(plan.product_mentions, chosen)
    if not_found:
        return {"error": "I couldn't find these in the catalog: " + ", ".join(not_found)
                         + ". Try the brand and model, e.g. 'Razer Blade 15'."}
    if needs_choice:
        return {"needs_choice": needs_choice}
    if plan.intent == "compare":
        if len(ids) < 2:
            return {"error": "Tell me at least two laptops to compare."}
        return _compare(plan, ids)
    if not ids:
        return {"error": "Which laptop do you mean?"}
    if plan.intent == "reviews":
        return _reviews(plan, ids[0])
    if plan.intent == "alternatives":
        return _alternatives(ids[0], budget)
    return {"error": "I'm not sure how to help with that yet."}
