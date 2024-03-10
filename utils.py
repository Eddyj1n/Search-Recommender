import nltk
import re
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    """
    Preprocesses the given text by lowercasing, removing special characters,
    tokenizing, lemmatizing, removing duplicates, and sorting words.
    """
    # Initialize the WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lowercase the text and remove special characters
    text_clean = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())

    # Tokenize and lemmatize each word
    words = text_clean.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    # Remove duplicate words 
    seen = set()
    unique_words = [x for x in lemmatized_words if not (x in seen or seen.add(x))]

    # Join the words back into a single string
    return ' '.join(unique_words)