import html, json, re, sqlite3
import pandas as pd

p = pd.read_csv("data/raw/products_raw.csv")
r = pd.read_csv("data/raw/reviews_raw.csv")

BRANDS = ["Acer", "ASUS", "Razer", "HP", "Lenovo", "Alienware", "Dell", "MSI",
          "Gigabyte", "Samsung", "Apple", "LG", "Microsoft", "Huawei", "Toshiba"]

def brand_of(title, store):
    for b in BRANDS:
        if re.search(rf"\b{b}\b", str(title), re.I):
            return b
    return str(store).strip() if pd.notna(store) else None

def first(pattern, text, flags=re.I):
    return re.search(pattern, str(text), flags)

def ram_gb(t):
    m = first(r"(\d{1,3})\s?GB\s*(?:DDR\d\w*|LPDDR\d\w*|RAM)", t)
    return int(m.group(1)) if m else None

def storage(t):
            m = first(r"(\d+(?:\.\d+)?)\s?(TB|GB)\s*(?:PCIe\s*|NVMe\s*|M\.2\s*|SATA\s*|SSD\s*)*Storage", t)
    if not m:
        m = first(r"(\d+(?:\.\d+)?)\s?(TB|GB)\s*Storage", t)
        if not m:
            return None, None
        gb = float(m.group(1)) * (1024 if m.group(2).upper() == "TB" else 1)
        return int(gb), None
    gb = float(m.group(1)) * (1024 if m.group(2).upper() == "TB" else 1)
    return int(gb), m.group(3).upper()

def cpu(t):
    m = first(r"(Core\s*Ultra\s*[579]\s*\w*|Core\s*i[3579][-\s]?\w+|Ryzen\s*[3579][-\s]?\w+|Apple\s*M\d\w*|Celeron\s*\w+|Pentium\s*\w+)", t)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    m = first(r"\b(i[3579][-+\s]?\d{4,5}\w*)", t)
    return m.group(1) if m else None

def gpu(t):
    m = first(r"((?:RTX|GTX)\s*\d{3,4}\s*(?:Ti|Super)?|\bMX\s?\d{3}|\bRX\s?\d{4}\w*)", t)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else None

def screen(t):
    m = first(r"(\d{2}(?:\.\d)?)\s*(?:\"|”|“|inch|-inch|in\b)", t)
    return float(m.group(1)) if m else None

def refresh(t):
    m = first(r"(\d{2,3})\s?Hz", t)
    return int(m.group(1)) if m else None

def os_of(t):
    m = first(r"Win(?:dows)?\s*(11|10)", t)
    return f"Windows {m.group(1)}" if m else None

def text_of(x):
    try:
        v = json.loads(x)
    except Exception:
        v = x
    if isinstance(v, list):
        v = " ".join(map(str, v))
    return re.sub(r"[\[\]]|\s+", " ", str(v)).strip()

rows = []
for i, row in enumerate(p.itertuples(index=False), start=1):
    t = row.title
    desc = (text_of(row.features) + " " + text_of(row.description)).strip()[:1500]
    st_gb, st_type = storage(t)
    if st_gb is None:
        st_gb, st_type = storage(desc)
    rows.append(dict(
        id=i, asin=row.parent_asin, name=t, brand=brand_of(t, row.store),
        price=float(row.price), rating=row.average_rating,
        rating_count=int(row.rating_number) if pd.notna(row.rating_number) else None,
        ram_gb=ram_gb(t) or ram_gb(desc), storage_gb=st_gb, storage_type=st_type,
        cpu=cpu(t) or cpu(desc), gpu=gpu(t) or gpu(desc),
        screen_in=screen(t) or screen(desc),
        refresh_hz=refresh(t) or refresh(desc),
        os=os_of(t) or os_of(desc), description=desc))
prod = pd.DataFrame(rows)

asin_to_id = dict(zip(prod["asin"], prod["id"]))

def clean_review(s):
    s = html.unescape(str(s))
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"\[\[[^\]]*\]\]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

r = r.copy()
r["text"] = r["text"].map(clean_review)
r["title"] = r["title"].fillna("").map(clean_review)
r = r[r["text"].str.len() >= 20].drop_duplicates(["parent_asin", "text"])
r["product_id"] = r["parent_asin"].map(asin_to_id)
r = r.dropna(subset=["product_id"])
rev = pd.DataFrame({
    "product_id": r["product_id"].astype(int), "rating": r["rating"],
    "title": r["title"], "text": r["text"],
    "helpful_votes": r["helpful_vote"].fillna(0).astype(int),
    "verified": r["verified_purchase"].astype(int), "ts": r["timestamp"]})
rev.insert(0, "id", range(1, len(rev) + 1))

con = sqlite3.connect("data/shop.db")
prod.to_sql("products", con, if_exists="replace", index=False)
rev.to_sql("reviews", con, if_exists="replace", index=False)
con.execute("CREATE INDEX IF NOT EXISTS idx_rev_pid ON reviews(product_id)")
con.commit()

print(len(prod), "products,", len(rev), "reviews")
print("\nSpec coverage (non-null out of", len(prod), "):")
print(prod[["brand","ram_gb","storage_gb","storage_type","cpu","gpu","screen_in","refresh_hz","os"]].notna().sum().to_string())
print("\nSample:")
print(prod[["id","brand","price","ram_gb","storage_gb","cpu","gpu","screen_in"]].head(8).to_string(index=False))