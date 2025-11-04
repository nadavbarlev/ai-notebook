from gensim.models import Word2Vec

# Tiny corpus with country–capital relationships
sentences = [
    "Paris is the capital of France".lower().split(),
    "Berlin is the capital of Germany".lower().split(),
    "Rome is the capital of Italy".lower().split(),
    "Madrid is the capital of Spain".lower().split(),
    "France has cities Paris and Lyon".lower().split(),
    "Germany has cities Berlin and Munich".lower().split(),
    "Italy has cities Rome and Milan".lower().split(),
    "Spain has cities Madrid and Barcelona".lower().split(),
    "Paris and Berlin are major european cities".lower().split(),
]

# Train Word2Vec on the tiny corpus
model = Word2Vec(sentences, vector_size=50, window=5, min_count=1, sg=1, epochs=50)

print("`France` is to `Paris` as `Germany` is to?")
word, _ = model.wv.most_similar(
    positive=["paris", "germany"], negative=["france"], topn=1
)[0]
print(word)
