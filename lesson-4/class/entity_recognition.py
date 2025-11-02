import spacy

# Load the small English model
# nlp = spacy.load("en_core_web_sm") # small model has limited vectors
nlp = spacy.load("en_core_web_md")

text = "Apple is looking at buying U.K. startup for $1 billion. I love processing text with spaCy!"

# Run the pipeline
doc = nlp(text)

# Tokens, lemmas, POS
print(f"{'Token':10} | {'Lemma':10} | {'POS':>3}")
for token in doc:
    print(f"{token.text:10} | {token.lemma_:10} | {token.pos_:>3}")

# Named Entities
print("\nNamed Entities:")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")

# Simple similarity (requires vectors; small model has limited vectors but still works)
sent1 = nlp("I enjoy natural language processing.")
sent2 = nlp("I like working with text.")
print("\nSimilarity between sentences:", round(sent1.similarity(sent2), 3))
