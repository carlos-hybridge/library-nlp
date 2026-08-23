from pydoc import doc

import spacy
import pandas as pd
from nltk.stem import SnowballStemmer


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


if __name__ == '__main__':
    main()
