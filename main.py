import spacy
import pandas as pd
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def tokenize(text):
    try:
        nlp = spacy.load('es_core_news_sm')
    except OSError:
        print("Downloading model...")
        from spacy.cli import download
        download('es_core_news_sm')
        nlp = spacy.load('es_core_news_sm')

    print(f"Texto cargado con 'exito. Longitud: {len(text)} caracteres.")

    doc = nlp(text)

    print(f"--- 1. Tokenización (Total tokens: {len(doc)}) ---")
    print([token.text for token in doc][:20])

    return doc


def filter_tokens(doc):
    relevant_tokens = []
    noise_toekns = []

    for token in doc:
        if token.is_stop and not token.is_punct and token.text.strip():
            relevant_tokens.append(token.text)
        elif token.is_stop:
            noise_toekns.append(token.text)
    print(f"\n--- 2. Filtrado de Stop Words ---")
    print(f"Palabras eliminadas (Ruido): {noise_toekns[:10]}...")
    print(f"Palabras conservadas (Contenido): {relevant_tokens[:10]}...")
    print(f"Reducción de tamaño: de {len(doc)} a {len(relevant_tokens)} tokens.")

    return relevant_tokens


def lemmatize(doc):
    normalized_tokens = []
    interesting_changes = []

    for token in doc:
        if not token.is_stop and not token.is_punct and token.text.strip():
            lemma = token.lemma_.lower()
            normalized_tokens.append(lemma)

            if token.text.lower() != lemma:
                interesting_changes.append(f"{token.text} -> {lemma}")

    print(f"\n--- 3. Lematización y Normalización ---")
    print(f"Total de tokens procesados: {len(normalized_tokens)}")
    print(f"Ejemplos de transformaciones (Palabra original ➡ Lema):")
    print(interesting_changes[:10])

    print(f"\nResultado final (Primeros 10 tokens):")
    print(normalized_tokens[:10])

    return normalized_tokens


def plot_words_3d(ax, matrix, vocabulary, title, points_color):
    words_matrix = matrix.T

    pca = PCA(n_components=3)
    coords = pca.fit_transform(words_matrix.toarray())

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    ax.scatter(x, y, z, c=points_color, s=80, edgecolors='k', alpha=0.8, depthshade=True)

    for i, word in enumerate(vocabulary):
        ax.text(x[i], y[i], z[i] + 0.1, word, fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("Comp. Principal 1")
    ax.set_ylabel("Comp. Principal 2")
    ax.set_zlabel("Comp. Principal 3")

    ax.plot([0,0], [0,0], [z.min(), z.max()], c='gray', linestyle='--', linewidth=0.5)
    ax.plot([x.min(), x.max()], [0,0], [0,0], c='gray', ls='--', lw=0.5, alpha=0.5)
    ax.plot([0,0], [y.min(), y.max()], [0,0], c='gray', ls='--', lw=0.5, alpha=0.5)


def main():
    with open("books/la_biblioteca_de_babel.txt", "r", encoding="utf-8") as f:
        text = f.read()

    stemmer = SnowballStemmer("spanish")
    comparative_data = []

    doc = tokenize(text)
    filter_tokens(doc)
    lemmatize(doc)

    for token in doc:
        if not token.is_punct and token.is_space:
            root_stem = stemmer.stem(token.text)
            lema = token.lemma_
            comparative_data.append({
                "Original": token.text,
                "Stemming (Corte)": root_stem,
                "Lematización (Diccionario)": lema,
                "¿Coinciden?": root_stem == lema
            })

    df = pd.DataFrame(comparative_data)

    print(f"\n--- 3. Stemming vs Lematización ---")
    interesting_words = ["hombres", "olvidado", "eres", "domesticado", "invisible", "ojos"]
    filter = df[df["Original"].isin(interesting_words)]

    print(filter.to_string(index=False))

    print("\n--- Visualización completa de los primeros 10 tokens ---")
    print(df.head(10).to_string(index=False))

    corpus_lemmatized = []

    for sentence in doc.sents:
        sentence_lemmas = [
            token.lemma_.lower()
            for token in sentence
            if not token.is_punct and not token.is_space and not token.is_stop
        ]
        if sentence_lemmas:
            corpus_lemmatized.append(" ".join(sentence_lemmas))

    print(f"Total de oraciones procesadas: {len(corpus_lemmatized)}")

    fig = plt.figure(figsize=(18, 8))

    ax1 = fig.add_subplot(121, projection='3d')

    bow_vectorizer = CountVectorizer()
    X_bow = bow_vectorizer.fit_transform(corpus_lemmatized)
    bow_vocabulary = bow_vectorizer.get_feature_names_out()

    plot_words_3d(ax1, X_bow, bow_vocabulary, "Espacio BoW 3D (Conteos)", "orange")

    ax2 = fig.add_subplot(122, projection='3d')

    tfidef_vectorizer = TfidfVectorizer()
    X_tfidf = tfidef_vectorizer.fit_transform(corpus_lemmatized)
    tfidf_vocabulary = tfidef_vectorizer.get_feature_names_out()

    plot_words_3d(ax2, X_tfidf, tfidf_vocabulary, "Espacio TF-IDF 3D (Importancia)", "teal")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
