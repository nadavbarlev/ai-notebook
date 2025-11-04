import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
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


text = "The data scientists are analyzing the datasets and building predictive models."

# Domain-specific corrections dictionary
# These handle special cases where lemmatization might not work correctly
domain_corrections = {
    "datasets": "dataset",
    "models": "model",
    "scientists": "scientist",
}

tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [
    lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tags
]

corrected_tokens = [
    domain_corrections.get(token.lower(), token) for token in lemmatized_tokens
]

print(f"{'Token':<15} | {'Lemmatized':<15} | {'Domain-Corrected':<15}")
print("-" * 60)
for index, token in enumerate(tokens):
    lemma = lemmatized_tokens[index]
    corrected = corrected_tokens[index]
    print(f"{token:<15} | {lemma:<15} | {corrected:<15}")
