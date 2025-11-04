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


def is_noun_or_verb(pos_tag):
    """
    Check if POS tag is a noun or verb.
    """
    return


def extract_keywords(text, use_lemmatization=True):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [
        token for token in tokens if token.isalpha() and token.lower() not in stop_words
    ]
    pos_tags = pos_tag(filtered_tokens)

    keywords = []
    for token, pos in pos_tags:
        if pos.startswith("N") or pos.startswith("V"):
            if use_lemmatization:
                lemmatizer = WordNetLemmatizer()
                keyword = lemmatizer.lemmatize(token.lower(), get_wordnet_pos(pos))
            else:
                keyword = token.lower()
            keywords.append(keyword)

    return keywords


text = """
The researchers are conducting experiments in multiple laboratories.
They conducted several studies last year and are conducting more this year.
The experiments involved testing various hypotheses across different laboratories.
Scientists published their findings after analyzing the data from these studies.
"""

tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)

# Extract keywords WITHOUT lemmatization
print("extracting keywords without lemmatization...")
keywords_no_lemma = extract_keywords(text, use_lemmatization=False)
print(f"keywords: {keywords_no_lemma}")
freq_no_lemma = Counter(keywords_no_lemma)
print(f"frequency: {freq_no_lemma}")
print()

print("extracting keywords with lemmatization...")
keywords_with_lemma = extract_keywords(text, use_lemmatization=True)
print(f"keywords: {keywords_with_lemma}")
freq_with_lemma = Counter(keywords_with_lemma)
print(f"frequency: {freq_with_lemma}")
print()
