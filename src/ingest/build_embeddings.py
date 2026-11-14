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