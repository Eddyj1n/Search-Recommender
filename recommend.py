import argparse
import pickle
import json
import warnings

from sklearn.metrics.pairwise import linear_kernel
from pathlib import Path
from keybert import KeyBERT

from utils import preprocess_text

# Suppress warnings
warnings.filterwarnings('ignore')

# Function to load a pickled object from file
def load_pickle(file_path):
    """
    Loads a pickled object from the specified file path.
    
    Args:
    file_path (str): Path to the pickled object file.

    Returns:
    object: The object loaded from the pickle file.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)   

# Function to extract keywords using KeyBERT
def get_keywords(text, query, max_keywords=3, model=None):
    """
    Extracts keywords from the given text using KeyBERT, excluding the query term.
    
    Args:
    text (str): The text to extract keywords from.
    query (str): The query term to exclude from the keywords.
    max_keywords (int): Maximum number of keywords to extract.
    model (KeyBERT model): The KeyBERT model to use for keyword extraction.

    Returns:
    str: A string of extracted keywords joined by space.
    """
    try:
        # Extract keywords
        keywords = model.extract_keywords(text, keyphrase_ngram_range=(1,1), stop_words="english", min_df=1)
        
        # Exclude the query term and select the top 'max_keywords' keywords
        top_keywords = [k[0] for k in keywords if k[0].lower() != query.lower()][:max_keywords]
        
        return " ".join(top_keywords)
    except Exception as e:
        print(f"Error in keyword extraction: {e}")
        return None

# Function to recommend search results based on a user search query
def get_recommendations(query, num_results, model, df, kw_model):
    """
    Recommends search results based on a user search query.

    Args:
    query (str): The search query.
    num_results (int): The number of results to return.
    model (TfidfVectorizer): The TF-IDF vectorizer model.
    df (pandas.Series): The series containing document data.
    kw_model (KeyBERT model): The KeyBERT model for keyword extraction.

    Returns:
    list: A list of recommended keywords.
    """
    # Preprocess the query based off same rules as IF-IDF model 
    processed_query = preprocess_text(query)
    
    # Extract up to 3 keywords from the processed query using KeyBERT.
    # KeyBERT is used to identify the most relevant keywords within the query.
    query_keywords = kw_model.extract_keywords(processed_query, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=3)

    # Create a list of main query terms. If KeyBERT extracts keywords successfully,
    # use them; otherwise, fall back to the original query.
    main_query_terms = [keyword[0] for keyword in query_keywords] if query_keywords else [query]

    # Combine the extracted keywords into a single string. 
    # the relevance of the search results.
    combined_query = ' '.join(main_query_terms)

    # Transform the combined query into its TF-IDF vector representation.
    query_tfidf = model.transform([combined_query])
    
    # Calculate cosine similarity between the query TF-IDF and the TF-IDF of all documents
    cosine_similarities = linear_kernel(query_tfidf, model.transform(df)).flatten()

    # Sort the documents based on their similarity to the query and get the indices of the top matches
    #related_docs_indices = cosine_similarities.argsort()[:-num_results-1:-1]
    
    related_docs_indices = cosine_similarities.argsort()[:-(num_results+1):-1]
    
    # Get the actual tags corresponding to the top matches
    related_topics = df.iloc[related_docs_indices].values.tolist()
    
    # Extract the keywords of the extracted topics, excluding the query term
    recommended_keywords = [get_keywords(topic, query, model=kw_model) for topic in related_topics]
    
    #Strip out first element if empty string 
    # if recommended_keywords[0] == '':
    #     recommended_keywords = recommended_keywords[1:num_results+1]
    # else:
    #     recommended_keywords = recommended_keywords[:num_results]
    
    return recommended_keywords

def interface():
    # Base path for script, data, and output directories
    base_path = Path(__file__).parent

    # Load models and data
    tfidf_vectorizer_artist = load_pickle(base_path / 'Output' / 'tfidf_vectorizer_arist_tag.pkl')
    artist_tag_series = load_pickle(base_path / 'Output' / 'artist_tags_series.pkl')
    
    # Initialize KeyBERT model
    kw_model = KeyBERT()

    # Argument parser for command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('query', type=str, help='Search query')
    parser.add_argument('--num_results', type=int, default=3, help='Number of results to return')
    args = parser.parse_args()

    if args.num_results < 1 or args.num_results > 15:
        print("Error: --num_results must be between 1 and 15")
    else:
        recommendations = get_recommendations(args.query, args.num_results, tfidf_vectorizer_artist, artist_tag_series, kw_model)
        print(json.dumps({"related_topics": recommendations}, indent=2))

if __name__ == "__main__":
    interface()
