# 4.
# import nltk
# from gensim.models import FastText, Word2Vec
# from nltk import pos_tag
# from nltk.corpus import stopwords, wordnet
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize

# nltk.download("punkt", quiet=True)
# nltk.download("stopwords", quiet=True)
# nltk.download("wordnet", quiet=True)
# nltk.download("averaged_perceptron_tagger", quiet=True)

# lemmatizer = WordNetLemmatizer()
# STOP = set(stopwords.words("english"))


# def wn_pos(tag):
#     return {
#         "J": wordnet.ADJ,
#         "V": wordnet.VERB,
#         "N": wordnet.NOUN,
#         "R": wordnet.ADV,
#     }.get(tag[0], wordnet.NOUN)


# def preprocess(s):
#     tokens = word_tokenize(s.lower())
#     tagged = pos_tag(tokens)
#     final_preprocess = [
#         lemmatizer.lemmatize(t, wn_pos(p))
#         for t, p in tagged
#         if t.isalpha() and t not in STOP
#     ]
#     print("final preprocessed tokens:", final_preprocess)
#     return final_preprocess


# corpus = [
#     "I love NLP.",
#     "Natural language processing is fun.",
#     "Learning about language helps.",
# ]
# docs = [preprocess(s) for s in corpus]

# # Train embeddings
# w2v = Word2Vec(docs, vector_size=30, window=2, min_count=1, sg=1)
# ft = FastText(docs, vector_size=30, window=2, min_count=1, sg=1)

# # Similar to "language"
# print("Word2Vec similar to 'language':", w2v.wv.most_similar("language", topn=2))
# print("FastText similar to 'language':", ft.wv.most_similar("language", topn=2))

# # Typo case
# typo = "lanquage"
# print("\nWord2Vec has typo?", typo in w2v.wv.key_to_index)
# try:
#     print("Word2Vec similarity typo->language:", w2v.wv.similarity(typo, "language"))
# except KeyError:
#     print("Word2Vec: typo out of vocabulary")

# print("FastText similarity typo->language:", ft.wv.similarity(typo, "language"))


# 3.
# import gensim.downloader as api
# import nltk
# from nltk.corpus import wordnet  # synonym database
# from scipy.spatial.distance import cosine

# nltk.download("wordnet")

# model = api.load("glove-wiki-gigaword-50")  # loading the vector model

# def top_synonyms(word, n=5):
#     lemmas = set(
#         lemma.name().replace("_", " ")
#         for syn in wordnet.synsets(word)
#         for lemma in syn.lemmas()
#         if lemma.name().lower() != word.lower()
#     )
#     print("lemmas:", lemmas)

#     candidates = [word for word in lemmas if word in model]
#     print("candidates:", candidates)

#     scored = sorted(
#         [
#             (candidate, 1 - cosine(model[candidate], model[word]))
#             for candidate in candidates
#         ],
#         key=lambda x: -x[1],
#     )
#     print("scored:", scored)
#     return scored[:n]

# print("top synonyms for 'happy':", top_synonyms("happy"))


# 2.
# import nltk
# from gensim.models import Word2Vec
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# nltk.download("punkt_tab", quiet=True)
# nltk.download("stopwords", quiet=True)

# sentences = [
#     "I love natural language processing.",
#     "Language models capture semantic meaning.",
#     "Word embeddings help with similarity tasks.",
#     "Deep learning creates vector representations of words.",
#     "We can find similar words using Word2Vec",
# ]

# tokenized = [
#     [
#         word
#         for word in word_tokenize(sentence.lower())
#         if word.isalpha() and word not in stopwords.words("english")
#     ]
#     for sentence in sentences
# ]

# # Training the model with the tokenized words
# # vector_size: The dimension of the feature vectors.
# # window: The maximum distance between the current and predicted word within a sentence.
# # min_count: The minimum number of times a word must appear in the training data to be included in the vocabulary.
# # sg: The training algorithm, 1 for skip-gram model, 0 for CBOW model.
# model = Word2Vec(tokenized, vector_size=50, window=2, min_count=1, sg=1) # building the vector model

# print("words most similar to 'language':", model.wv.most_similar("language", topn=3))
# print("words most similar to 'deep':", model.wv.most_similar("deep", topn=3))


# 1.
# import gensim.downloader as api

# # 50-dimensional vectors: Each word is represented as a 50-element vector
# # Training data: Trained on Wikipedia + Gigaword 5 corpora (approximately 6 billion tokens)
# # Purpose: Captures semantic (meaning) relationships between words (a vector analogy)
# model = api.load(
#     "glove-wiki-gigaword-50"
# )  # this is a one‑time download + cache locally

# # Direct similarity with score
# king_queen_similarity = model.similarity("king", "queen")
# print(f"king ~ queen: {king_queen_similarity:.4f}")

# # Analogy with scores
# analogy_results = model.most_similar(
#     positive=["paris", "germany"],
#     negative=["france"],
#     topn=10,
# )

# print("paris + germany - france →")
# for word, score in analogy_results:
#     print(f"  {word}: {score:.4f}")
