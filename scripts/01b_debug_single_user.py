"""Script 01b — debug runner reproducing one test user's pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from cs_core.blend import EmbeddingBlend

DATA = Path("ml-latest-small")
ratings = pd.read_csv(DATA / "ratings.csv")
movies = pd.read_csv(DATA / "movies.csv")
pos = ratings[ratings["rating"] >= 4.0]
npos = pos.groupby("userId").size()
warm = npos[npos >= 10].index.tolist()
print("warm:", len(warm))

# train quick item2vec (same params as 01)
seqs = pos.groupby("userId")["movieId"].apply(list).tolist()
seqs = [s for s in seqs if len(s) >= 3]
model = Word2Vec(seqs, vector_size=128, window=5, min_count=2, sg=1, epochs=5, workers=1, seed=42)
print("vocab:", len(model.wv))

movie_genre = dict(zip(movies["movieId"], movies["genres"].fillna("")))
user_items = pos.groupby("userId")["movieId"].apply(set).to_dict()


def genre_of(ids):
    g = set()
    for m in ids:
        g.update(movie_genre.get(m, "").split("|"))
    return g


def user_emb(model, ids):
    vecs = [model.wv[m] for m in ids if m in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)


# candidate pool: ids that exist in the model (keys are int)
cand_ids = [k for k in model.wv.key_to_index if isinstance(k, int)][:3000]

uid = warm[0]
print("test user:", uid)
held = set(user_items[uid])
all_ids = sorted(held)
n_held = max(1, int(0.2 * len(all_ids)))
held_ids = set(all_ids[-n_held:])
train_ids = [m for m in all_ids if m not in held_ids]
print("train:", len(train_ids), "held:", len(held_ids))
u_emb = user_emb(model, train_ids)
print("u_emb nonzero:", np.any(u_emb))

best = max(
    (m for m in warm if m != uid),
    key=lambda m: len(genre_of(user_items[m]) & genre_of(train_ids)),
)
print("mentor:", best, "genre-overlap:", len(genre_of(user_items[best]) & genre_of(train_ids)))
mentor_ids = [m for m in user_items[best] if m not in held]
m_emb = user_emb(model, mentor_ids[:60])
print("m_emb nonzero:", np.any(m_emb))

cand_ids = [m for m in cand_ids if m not in train_ids]
C = np.array([model.wv[m] for m in cand_ids])
rel = np.array([1 if m in held_ids else 0 for m in cand_ids])
print("cands:", len(cand_ids), "relevant:", int(rel.sum()))

base_scores = C @ u_emb
blend = EmbeddingBlend()
for a in (0.2, 0.45, 0.7):
    b = blend.blend(u_emb, [m_emb], a)
    bs = C @ b
    print(f"alpha={a}: corr(base,blend)={np.corrcoef(base_scores, bs)[0,1].round(3)}")
print("done single-user debug")