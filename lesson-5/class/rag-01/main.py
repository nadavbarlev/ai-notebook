import os

import faiss
import nltk
import numpy as np
from google import genai
from google.genai import types
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer

client = genai.Client(api_key="AIzaSyCfDYXwz8B9_3jC2yrk7cUQQbpG2k1CTLg")

nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


# Load and preprocess documents
def load_documents(folder="."):
    all_sentences = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                text = f.read()
                for sentence in sent_tokenize(text):  # tokenize into sentences
                    words = word_tokenize(sentence)  # tokenize into words
                    clean = [
                        w for w in words if w.lower() not in stop_words and w.isalnum()
                    ]
                    all_sentences.append(" ".join(clean))
    return all_sentences


# Create FAISS index
def create_faiss(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(
        dim
    )  # exact (brute-force) nearest-neighbor index using L2 distance.
    index.add(np.array(embeddings))
    return index  # like `collection` in vector database


# Retrieve top-k sentences
def retrieve(query, model, faiss_index, sentences, k=3):
    query_embedding = model.encode([query])
    _, indices = faiss_index.search(
        np.array(query_embedding), k
    )  # FAISS preserving the order of the added embeddings
    return [sentences[i] for i in indices[0]]


def ask_gemini(context, question):
    prompt = f"""
Use the following context to answer the question clearly:
\ncontext:\n{context}
\nquestion:\n{question}
\nanswer:
"""
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=1
            )  # Disable "thinking"
        ),
    )
    return response.text.strip()


def main():
    print("loading documents...")
    tokenized_sentences = load_documents()

    print("loading sentence transformer model...")
    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )  # the model maps any sentence to a fixed-size dense vector

    print("creating embeddings...")
    embeddings = model.encode(tokenized_sentences)

    print("creating faiss index...")
    faiss_index = create_faiss(embeddings)  # facebook ai's similarity search

    while True:
        q = input("\nask something (or type 'exit' to quit): ")
        if q.lower() == "exit":
            break

        top_chunks = retrieve(q, model, faiss_index, tokenized_sentences)
        context = "\n".join(top_chunks)
        print("\nretrieved context:\n", context)

        answer = ask_gemini(context, q)
        print("\ngemini answer:\n", answer)


if __name__ == "__main__":
    main()
