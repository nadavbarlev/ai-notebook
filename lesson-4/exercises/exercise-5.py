from itertools import combinations

from gensim.models import Word2Vec

# Mini corpus
tokens = [
    "king",
    "queen",
    "man",
    "woman",
    "prince",
    "princess",
]

# Train a very small Word2Vec model
model = Word2Vec([tokens], vector_size=20, window=2, min_count=1, sg=1, epochs=200)

pairs_to_check = [("king", "queen"), ("man", "woman"), ("king", "man")]
for a, b in pairs_to_check:
    sim = model.wv.similarity(a, b)
    print(f"  {a} ~ {b}: {sim:.4f}")

print("\nall pair similarities (sorted):")
all_words = tokens
pair_sims = []
for a, b in combinations(all_words, 2):
    pair_sims.append(((a, b), float(model.wv.similarity(a, b))))

pair_sims.sort(key=lambda x: x[1], reverse=True)
for (a, b), s in pair_sims:
    print(f"  {a} ~ {b}: {s:.4f}")
