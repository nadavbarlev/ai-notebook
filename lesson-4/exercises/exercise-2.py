import nltk
import spacy
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

# Load French spaCy model
try:
    nlp_fr = spacy.load("fr_core_news_sm")
except OSError:
    print("Downloading fr_core_news_sm...")
    import subprocess

    subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_sm"], check=True)
    nlp_fr = spacy.load("fr_core_news_sm")


def get_wordnet_pos(treebank_tag):
    """Convert NLTK POS tag to WordNet POS tag format."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


english_text = "She was running quickly and finished early."
french_text = "Elle courait rapidement et a terminé tôt."

print("english lemmatization with nltk wordnet lemmatizer:")
print(f"Text: {english_text}\n")

tokens_en = word_tokenize(english_text)
pos_tags_en = pos_tag(tokens_en)
lemmatizer = WordNetLemmatizer()

print(f"{'Token':<12} | {'POS':<8} | {'Lemma':<12}")
print("-" * 35)
for token, pos in pos_tags_en:
    if token.isalpha():
        lemma = lemmatizer.lemmatize(token, get_wordnet_pos(pos))
        print(f"{token:<12} | {pos:<8} | {lemma:<12}")

print("")

# French lemmatization with spaCy
print("french lemmatization with spacy lemmatizer:")
print(f"Text: {french_text}\n")

doc_fr = nlp_fr(french_text)

print(f"{'Token':<12} | {'POS':<8} | {'Lemma':<12}")
print("-" * 35)
for token in doc_fr:
    if not token.is_punct and not token.is_space:
        print(f"{token.text:<12} | {token.pos_:<8} | {token.lemma_:<12}")
