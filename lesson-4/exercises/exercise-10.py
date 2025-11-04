import nltk
import numpy as np
from gensim.models import Word2Vec
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)


def get_wordnet_pos(treebank_tag):
    """Convert NLTK POS tag to WordNet POS tag format for better lemmatization."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess_text(text):
    """Preprocess text: tokenization, stopword removal, and lemmatization."""

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword removal and filter alphabetic tokens only
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [
        token for token in tokens if token.isalpha() and token.lower() not in stop_words
    ]

    # POS tagging
    pos_tags = pos_tag(filtered_tokens)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmas = [
        lemmatizer.lemmatize(token.lower(), get_wordnet_pos(pos))
        for token, pos in pos_tags
    ]

    return lemmas


def sentence_to_vector(lemmas, model):
    """Generate sentence embedding by averaging Word2Vec vectors of words."""
    vectors = []
    for lemma in lemmas:
        if lemma in model.wv.key_to_index:
            vectors.append(model.wv[lemma])

    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


texts = [
    "Deep learning improves image recognition accuracy.",
    "Stock predictions rely on time-series analysis.",
    "Language models are used in machine translation.",
]

print("preprocessing texts...")
preprocessed = []
for i, text in enumerate(texts, 1):
    lemmas = preprocess_text(text)
    preprocessed.append(lemmas)
    print(f"text {chr(64 + i)}: {text}")
    print(f"preprocessed: {lemmas}")
    print()

print("training word2vec model...")
model = Word2Vec(
    preprocessed,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    epochs=100,
)
print(f"vocabulary: {list(model.wv.key_to_index.keys())}")
print(f"vocabulary size: {len(model.wv.key_to_index)} words")
print()

print("generating sentence embeddings...")
sentence_vectors = []
for i, lemmas in enumerate(preprocessed, 1):
    vec = sentence_to_vector(lemmas, model)
    sentence_vectors.append(vec)
    print(f"text {chr(64 + i)} embedding shape: {vec.shape}")
print()

sentence_vectors = np.array(sentence_vectors)
print(f"all embeddings shape: {sentence_vectors.shape}")
print()

print("clustering with k-means...")
for n_clusters in [2, 3]:
    print(f"k-means with {n_clusters} clusters:")

    # random_state: fixes the random seed for reproducible results.
    # n_init: number of times the k-means algorithm will be run with different centroid seeds
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(sentence_vectors)

    # Group texts by cluster
    cluster_groups = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(chr(65 + idx))

    for cluster_id in sorted(cluster_groups.keys()):
        texts_in_cluster = cluster_groups[cluster_id]
        print(f"cluster {cluster_id}: {', '.join(texts_in_cluster)}")
        for text_id in texts_in_cluster:
            idx = ord(text_id) - 65
            print(f"- text {text_id}: {texts[idx]}")

    print()
