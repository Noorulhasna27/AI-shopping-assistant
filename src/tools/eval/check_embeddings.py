import numpy as np
from fastembed import TextEmbedding
from sentence_transformers import SentenceTransformer

fast = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
st = SentenceTransformer("all-MiniLM-L6-v2")
for q in ["battery drains fast", "gets hot and loud", "great screen for the price",
          "best gaming laptop for a student"]:
    a = np.asarray(next(iter(fast.embed([q]))), dtype="float32")
    a /= np.linalg.norm(a)
    b = st.encode([q], normalize_embeddings=True)[0]
    print(f"{q!r}: cosine = {float(a @ b):.4f}")