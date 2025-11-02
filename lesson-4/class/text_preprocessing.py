# 8.
# import nltk
# from nltk.tokenize import word_tokenize

# nltk.download("punkt", quiet=True)


# def tokenize(text):
#     return [t.lower() for t in word_tokenize(text) if t.isalpha()]


# def common_word_count(s1, s2):
#     return len(set(tokenize(s1)) & set(tokenize(s2)))


# def percentage_overlap(s1, s2):
#     t1 = set(tokenize(s1))
#     t2 = set(tokenize(s2))
#     if not (t1 | t2):  # all unique elements from both sets
#         return 0.0
#     return len(t1 & t2) / len(t1 | t2) * 100.0


# a = "Natural language processing is fun!"
# b = "I find processing natural language quite enjoyable."

# print("tokens a:", tokenize(a))
# print("tokens b:", tokenize(b))
# print("shared count:", common_word_count(a, b))
# print(f"overlap %: {percentage_overlap(a, b):.1f}%")


# 7. (!)
# import nltk
# from nltk import pos_tag, word_tokenize
# from nltk.corpus import wordnet
# from nltk.stem import PorterStemmer, WordNetLemmatizer

# nltk.download("punkt_tab", quiet=True)
# nltk.download("wordnet", quiet=True)
# nltk.download("omw-1.4", quiet=True)
# nltk.download("averaged_perceptron_tagger_eng", quiet=True)

# # Helpers to map NLTK POS tags to WordNet's format
# def get_wordnet_pos(treebank_tag):
#     if treebank_tag.startswith("J"):
#         return wordnet.ADJ
#     if treebank_tag.startswith("V"):
#         return wordnet.VERB
#     if treebank_tag.startswith("N"):
#         return wordnet.NOUN
#     if treebank_tag.startswith("R"):
#         return wordnet.ADV
#     return wordnet.NOUN  # fallback


# text = "The striped bats are hanging on their feet and they are better than before."

# tokens = word_tokenize(text)
# pos_tags = pos_tag(tokens)  # part of speech tagging
# print("pos_tags:", pos_tags)

# stemmer = (
#     PorterStemmer()
# )  # stemming: reducing a word to its root form by removing suffixes
# lemmatizer = (
#     WordNetLemmatizer()
# )  # lemmatization: reducing a word to its base form more contextually correct

# print(f"{'token':10} | {'stemmed':10} | {'lemmatized':10}")
# for token, pos in pos_tags:
#     stem = stemmer.stem(token)
#     lemma = lemmatizer.lemmatize(token, get_wordnet_pos(pos))
#     print(f"{token:10} | {stem:10} | {lemma}")


# 6.
# import nltk
# from nltk.tokenize import RegexpTokenizer, word_tokenize

# nltk.download("punkt_tab", quiet=True)

# text = "Children aren't playing soccer-they've gone inside."

# default_tokens = word_tokenize(text)
# print("default:", default_tokens)

# regexp_tokenizer = RegexpTokenizer(
#     r"\w+"
# )  # create different tokenizer which splits on any non-word or digit character
# tokens = regexp_tokenizer.tokenize(text)
# print("regex:", tokens)


# 5.
# import nltk
# from nltk.tokenize import word_tokenize

# nltk.download("punkt_tab", quiet=True)

# positive = {"good", "great", "happy", "fun", "love", "powerful"}
# negative = {"bad", "sad", "hate", "terrible", "hard"}

# def sentiment(text):
#     tokens = [t.lower() for t in word_tokenize(text)]
#     pos = sum(1 for t in tokens if t in positive)
#     neg = sum(1 for t in tokens if t in negative)
#     if pos > neg:
#         return "Positive"
#     elif neg > pos:
#         return "Negative"
#     else:
#         return "Neutral"

# print(sentiment("This is a great product!"))
# print(sentiment("This is a bad product!"))


# 4. (!)
# import nltk
# from nltk.corpus import stopwords

# nltk.download("punkt_tab", quiet=True)
# nltk.download("stopwords", quiet=True)

# text = "This is an example showing how tokenization and stopword removal work."
# tokens = [t.lower() for t in nltk.word_tokenize(text)]
# filtered = [
#     token
#     for token in tokens
#     if token.isalpha() and token not in stopwords.words("english")
# ]

# print(
#     "filtered tokens:", filtered
# )  # text preprocessing pipeline step before vectorization or modeling.


# 3.
# # from collections import Counter

# import nltk
# from nltk.tokenize import word_tokenize

# nltk.download("punkt", quiet=True)

# text = "Python is great. Python is simple. NLP with Python is powerful!"
# tokens = [t.lower() for t in word_tokenize(text)]
# freq = Counter(tokens)

# print("Frequencies:", freq)


# 2.
# import nltk

# nltk.download("punkt_tab", quiet=True)

# text = "Don't stop believing! it's amazing."
# print("split():", text.split())
# print("word_tokenize():", nltk.word_tokenize(text))


# 1.
# import nltk  # Natural Language Toolkit

# nltk.download("punkt_tab")  # Download the punkt_tab resource (run once)

# text = "Hello, how are you? I am fine, thank you."
# tokens = nltk.word_tokenize(
#     text
# )  # Tokens are the smallest units of a text that are meaningful

# print(tokens)
