from gensim.models import Word2Vec


def create_training_corpus(text):
    """
    Create a corpus suitable for Word2Vec training.
    Returns a list of tokenized sentences.
    """
    sentences = text.strip().split(".")
    corpus = []
    for sentence in sentences:
        if sentence.strip():
            tokens = [
                word.strip().lower()
                for word in sentence.split()
                if word.strip() and word.isalpha()
            ]
            if tokens:
                corpus.append(tokens)
    return corpus


def train_word2vec_models(corpus, window_sizes):
    """
    Train Word2Vec models with different context window sizes.

    Args:
        corpus: List of tokenized sentences
        window_sizes: List of window sizes to try

    Returns:
        Dictionary mapping window size to trained model
    """
    models = {}
    for window_size in window_sizes:
        print(f"training word2vec model with window size: {window_size}")
        model = Word2Vec(
            sentences=corpus,
            vector_size=100,  # Dimensionality of word vectors
            window=window_size,  # Context window size
            min_count=1,  # Minimum word frequency
            workers=1,  # Number of threads
            sg=1,  # Use Skip-gram (1) instead of CBOW (0)
            epochs=50,  # Number of training iterations over the corpus
            seed=42,  # For reproducibility
        )
        models[window_size] = model

    return models


def compute_similarity(model, word1, word2):
    try:
        similarity = model.wv.similarity(word1, word2)
        return similarity
    except KeyError as e:
        print(f"warning: word not in vocabulary: {e}")
        return None


def analyze_word_relationships(models, word_pairs):
    """
    Analyze how word relationships change across different window sizes.

    Args:
        models: Dictionary of window_size -> Word2Vec model
        word_pairs: List of tuples containing word pairs to analyze
    """
    for word1, word2 in word_pairs:
        print(f"analyzing relationship: '{word1}' <-> '{word2}'")
        similarities = []

        for window_size in sorted(models.keys()):
            model = models[window_size]
            similarity = compute_similarity(model, word1, word2)

            if similarity is not None:
                similarities.append((window_size, similarity))
                print(f"window size {window_size:2d}: similarity = {similarity:.4f}")

        if len(similarities) >= 2:
            trend = (
                "increasing"
                if similarities[-1][1] > similarities[0][1]
                else "decreasing"
            )
            change = abs(similarities[-1][1] - similarities[0][1])
            print(f"  → Trend: {trend} (change: {change:.4f})")


def find_most_similar_words(models, target_word, top_n=5):
    """
    Find most similar words for a target word across different window sizes.

    Args:
        models: Dictionary of window_size -> Word2Vec model
        target_word: The word to find similarities for
        top_n: Number of most similar words to return
    """
    print(f"\n--- Most similar words to '{target_word}' ---")

    for window_size in sorted(models.keys()):
        model = models[window_size]
        try:
            similar_words = model.wv.most_similar(target_word, topn=top_n)
            print(f"\n  Window size {window_size}:")
            for word, similarity in similar_words:
                print(f"    {word:15s} (similarity: {similarity:.4f})")
        except KeyError:
            print(f"\n  Window size {window_size}: '{target_word}' not in vocabulary")


text = """
The doctor prescribed medicine to the patient who was recovering in the hospital.
The patient visited the doctor at the hospital yesterday.
The hospital has many doctors who treat patients daily.
A doctor examined the patient and recommended medicine.
The medicine helped the patient recover faster in the hospital.
Doctors and nurses work together in the hospital to help patients.
The patient thanked the doctor for the excellent medicine and care.
The doctor carefully reviewed the patient's medical history.
Patients often visit the hospital for routine checkups with their doctor.
The hospital emergency room was busy with patients needing immediate care.
A skilled doctor can diagnose patients quickly and prescribe the right medicine.
The patient felt better after taking the medicine the doctor recommended.
Hospital staff including doctors and nurses provide excellent patient care.
The doctor explained to the patient how the medicine works in the body.
Many patients arrive at the hospital seeking help from experienced doctors.
The medicine needs to be taken as prescribed by the doctor for the patient.
Doctors at the hospital work around the clock to help patients in need.
The patient asked the doctor about potential side effects of the medicine.
The hospital pharmacy dispenses medicine that doctors prescribe for patients.
A good doctor listens carefully to patient concerns and provides medicine accordingly.
Patients recovering in the hospital benefit from attentive care by doctors.
The doctor adjusted the medicine dosage based on the patient's condition.
Hospital doctors collaborate to provide comprehensive treatment for all patients.
The medicine proved effective and the patient's condition improved significantly.
Doctors use their expertise to choose the best medicine for each patient.
The patient was grateful for the doctor's care and the helpful medicine.
Hospital administrators ensure doctors have the resources needed to treat patients.
The doctor monitored the patient closely after administering the new medicine.
Patients trust their doctors to prescribe medicine that will help them recover.
The hospital maintains high standards so doctors can provide quality patient care.
"""

print("preparing corpus...")
corpus = create_training_corpus(text)

print("training word2vec models...")
window_sizes = [2, 5, 10]
models = train_word2vec_models(corpus, window_sizes)

print("analyzing word relationships...")
word_pairs = [
    ("doctor", "patient"),
    ("doctor", "hospital"),
    ("patient", "hospital"),
    ("medicine", "patient"),
    ("doctor", "medicine"),
]
analyze_word_relationships(models, word_pairs)
