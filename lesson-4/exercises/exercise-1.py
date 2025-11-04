import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)


# Helper function to map NLTK POS tags to WordNet's format
def get_wordnet_pos(treebank_tag):
    """
    Convert NLTK POS tag to WordNet POS tag format.
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


text = (
    "The children were playing in the park while their parents watched from the bench."
)

tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)

lemmatizer = WordNetLemmatizer()
lemmas_without_pos = [lemmatizer.lemmatize(token) for token in tokens]
lemmas_with_pos = [
    lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tags
]

print(f"{'Token':12} | {'POS':12} | {'Without POS':15} | {'With POS':15}")
for i, (token, pos_tag_item) in enumerate(pos_tags):
    without_pos = lemmas_without_pos[i]
    with_pos = lemmas_with_pos[i]
    print(f"{token:<14} {pos_tag_item:<14} {without_pos:<17} {with_pos:<17}")
