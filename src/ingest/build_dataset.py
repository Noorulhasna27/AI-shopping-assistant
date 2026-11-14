import json, os, pandas as pd, pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download

os.makedirs("data/raw", exist_ok=True)

rev = load_dataset("naga-jay/amazon-laptop-gpu-reviews", split="train").to_pandas()
rev = rev[rev["text"].str.len() >= 40]
counts = rev.groupby("parent_asin").size().sort_values(ascending=False)
print(len(rev), "reviews |", len(counts), "products |", (counts >= 20).sum(), "with 20+ reviews")
cand = counts[counts >= 15].head(300).index.tolist()

COLS = ["parent_asin","title","price","average_rating","rating_number",
        "store","features","description","details","categories"]
parts = []
for i in range(10):
    p = hf_hub_download("McAuley-Lab/Amazon-Reviews-2023", repo_type="dataset",
        filename=f"raw_meta_Electronics/full-{i:05d}-of-00010.parquet",
        revision="refs/pr/10")
    t = pq.read_table(p, columns=COLS, filters=[("parent_asin", "in", cand)])
    parts.append(t.to_pandas())
    print("shard", i, "->", len(parts[-1]))
meta = pd.concat(parts, ignore_index=True)

def to_price(x):
    try: return float(str(x).replace("$", "").replace(",", ""))
    except Exception: return None

meta["price"] = meta["price"].map(to_price)
meta = meta[meta["price"].notna() & meta["title"].notna()].drop_duplicates("parent_asin")
print(len(meta), "products with price + title")
print(meta[["title", "price", "rating_number"]].head(10).to_string())

n = rev.groupby("parent_asin").size()
keep = n[n.index.isin(meta["parent_asin"]) & (n >= 20)].sort_values(ascending=False).head(80).index
products = meta[meta["parent_asin"].isin(keep)].copy()
for c in ["features", "description", "details", "categories"]:
    products[c] = products[c].map(lambda x: json.dumps(x, default=str))
reviews = (rev[rev["parent_asin"].isin(keep)]
           .sort_values("helpful_vote", ascending=False)
           .groupby("parent_asin").head(40)
           [["parent_asin","rating","title","text","helpful_vote","verified_purchase","timestamp"]])
print(len(products), "products,", len(reviews), "reviews")

products.to_csv("data/raw/products_raw.csv", index=False)
reviews.to_csv("data/raw/reviews_raw.csv", index=False)

