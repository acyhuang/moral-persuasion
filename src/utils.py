import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)

# Load Porter Stemmer
porterStemmer = PorterStemmer()

def stem_sentences(sentences):
    """Stem a set of sentences"""
    sentences_tokenized = [word_tokenize(sentence) for sentence in sentences]

    sentences_stemmed = []
    for sentence_tokenized in sentences_tokenized:
        sentences_stemmed.append(
            " ".join(porterStemmer.stem(token) for token in sentence_tokenized)
        )

    return sentences_stemmed