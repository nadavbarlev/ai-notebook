from collections import Counter

import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)


# Helper function to map NLTK POS tags to WordNet's format
def get_wordnet_pos(treebank_tag):
    """
    Convert NLTK POS tag to WordNet POS tag format for better lemmatization.
    """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN  # fallback to noun if tag doesn't match


text = "Dogs are running faster than the cats that were sleeping under the table."

# Tokenization
tokens = word_tokenize(text)
print(tokens)
print()

# Stopword
stop_words = set(stopwords.words("english"))
filtered_tokens = [
    token.lower()
    for token in tokens
    if token.isalpha() and token.lower() not in stop_words
]
print(filtered_tokens)
print()

# Lemmatization
lemmatizer = WordNetLemmatizer()
pos_tags = pos_tag(
    [token for token in tokens if token.isalpha() and token.lower() not in stop_words]
)  # We need to tag the original case tokens for better accuracy
lemmas = [
    lemmatizer.lemmatize(token.lower(), get_wordnet_pos(pos)) for token, pos in pos_tags
]
print(lemmas)
print()

# Frequency distribution
freq_dist = Counter(lemmas)
print(freq_dist)
print()
