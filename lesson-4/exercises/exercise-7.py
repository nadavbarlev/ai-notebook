import nltk
import numpy as np
from gensim import downloader
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess_sentence(sentence):
    tokens = word_tokenize(sentence)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [
        token for token in tokens if token.isalpha() and token.lower() not in stop_words
    ]
    pos_tags = pos_tag(filtered_tokens)
    lemmatizer = WordNetLemmatizer()
    lemmas = [
        lemmatizer.lemmatize(token.lower(), get_wordnet_pos(pos))
        for token, pos in pos_tags
    ]
    return lemmas


def sentence_to_vector(sentence_lemmas, model):
    vectors = []
    for lemma in sentence_lemmas:
        if lemma in model.key_to_index:
            vectors.append(model[lemma])
    if len(vectors) == 0:
        return None
    return np.mean(vectors, axis=0)  # Return average vector


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


sentence1 = "The man is playing guitar."
sentence2 = "A person performs music on stage."

print("preprocessing sentences...")
lemmas1 = preprocess_sentence(sentence1)
lemmas2 = preprocess_sentence(sentence2)

print("loading model...")
model = downloader.load("glove-wiki-gigaword-50")
print(f"vocabulary size: {len(model.key_to_index):,} words")

print("computing average word vectors...")
vec1 = sentence_to_vector(lemmas1, model)
vec2 = sentence_to_vector(lemmas2, model)

if vec1 is None or vec2 is None:
    print("error: could not compute vectors. some words may not be in the vocabulary.")
    exit(1)

print("calculating cosine similarity...")
similarity = cosine_similarity(vec1, vec2)
print(f"cosine similarity: {similarity:.4f}")
