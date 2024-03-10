import pandas as pd
import nltk
import pickle
import warnings
import unittest
import json
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from keybert import KeyBERT
from io import StringIO
from unittest.mock import patch
from unittest.mock import MagicMock

from utils import preprocess_text
from recommend import load_pickle, get_keywords, get_recommendations, interface

# Suppress warnings
warnings.filterwarnings('ignore')

#Ensure required NLTK data is downloaded
nltk.download('wordnet')

def load_stop_words(file_path):
    """
    Loads a list of stop words from the specified file.
    """
    # Read stop words from the given file
    with open(file_path, 'r') as file:
        return file.read().splitlines()

def read_and_preprocess_data(artist_tags_path, user_search_path):
    """
    Reads and preprocesses the artist tags and user search data.
    """
    # Read artist tags and user search data from CSV and TSV files
    artist_tags_df = pd.read_csv(artist_tags_path, header=None, names=['tags'])
    user_search_df = pd.read_csv(user_search_path, delimiter='\t', usecols=['keywords'])

    # Drop NaN values from both dataframes
    artist_tags_df.dropna(inplace=True)
    user_search_df.dropna(inplace=True)

    # Preprocess tags and keywords by splitting and joining them
    artist_tags_df['tags'] = artist_tags_df['tags'].apply(lambda x: ' '.join(x.split(',')))
    user_search_df['keywords'] = user_search_df['keywords'].apply(lambda x: ' '.join(x.split(',')))

    # Apply the text preprocessing function to both dataframes
    artist_tags_series = artist_tags_df['tags'].apply(preprocess_text).drop_duplicates()
    user_search_series = user_search_df['keywords'].apply(preprocess_text).drop_duplicates()

    return artist_tags_series, user_search_series

def main():
    # Base path for script, data, and output directories
    base_path = Path(__file__).parent

    # Construct file paths relative to the base path
    artist_tags_path = base_path / 'Data' / 'artist_tags.csv'
    user_search_path = base_path / 'Data' / 'user_search_keywords.tsv'
    stop_words_path = base_path / 'aux' / 'stop_words.txt'
    output_path = base_path / 'Output'
    
    # Create output directory if it doesn't exist
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # Load custom stop words
    custom_stop_words = load_stop_words(stop_words_path)

    # Read and preprocess data
    artist_tags_series, user_search_series = read_and_preprocess_data(artist_tags_path, user_search_path)

    # Create and configure the TF-IDF vectorizer
    tfidf_vectorizer_artist_tag = TfidfVectorizer(
        stop_words=custom_stop_words,
        analyzer='word',
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.6
    )

    # Fit the TF-IDF vectorizer on the artist tags data
    tfidf_vectorizer_artist_tag.fit(artist_tags_series)

    # Save the TF-IDF model and preprocessed data
    with open(output_path / 'tfidf_vectorizer_arist_tag.pkl', 'wb') as file:
        pickle.dump(tfidf_vectorizer_artist_tag, file)

    with open(output_path / 'artist_tags_series.pkl', 'wb') as file:
        pickle.dump(artist_tags_series, file)

    with open(output_path / 'user_search_series.pkl', 'wb') as file:
        pickle.dump(user_search_series, file)
        
# Create class for unit tests
class tests(unittest.TestCase): 
    
    #Create method to check pre-process function is working as intended
    def test_preprocess_text(self): 
        
        #Check lower casing function works
        self.assertEqual(preprocess_text("This is a test!"), "this is a test") 
        
        #Check special character removal works
        self.assertEqual(preprocess_text("123 Test@!$#$!"), "123 test")
        
        #check tokenization and removal fo duplicates works
        self.assertEqual(preprocess_text("123 Test@!$#$! Test@!$#$! test"), "123 test")
    
    #Set up mock data and objects for testing
    def setUp(self):
        
        #Test number of results returned when specicific numeric argument parsed 
        self.mock_num_results = 5
        
        #Create mock data and firt mock model 
        sample_data = ["sample tag 1", "sample tag 2", "sample tag 3", "sample tag 4", "sample tag 5", 
                       "sample tag 6", "sample tag 7", "sample tag 8", "sample tag 9", "sample tag 10",
                       "sample tag 11", "sample tag 12", "sample tag 13", "sample tag 14", "sample tag 15"]
        
        #Initiate mock vectorizer model
        self.mock_model = TfidfVectorizer()
        
        #Fit model to sample data 
        self.mock_model.fit(sample_data)
        
        #Save mock data as series object
        self.mock_df =pd.Series(sample_data)
        
        #Initate keybert model
        #self.mock_kw_model = KeyBERT()
        self.mock_kw_model = MagicMock()

     # Create test method to check the get_recommendations function output is correct
    def test_recommendations_ouput(self): 
         
        #Test with default number of results
        recommendations= get_recommendations("test", self.mock_num_results, self.mock_model, self.mock_df, self.mock_kw_model)
        
        #Test if result matches user set argument
        self.assertEqual(len(recommendations), self.mock_num_results)
        
        #Test if the output is a list
        self.assertIsInstance(recommendations, list)
        
     # Test to ensure that the interface function outputs valid JSON.   
    def test_interface_json_output(self):

        # Simulate command-line arguments
        test_args = ["script_name", "test query", "--num_results", "3"]
        
        with patch.object(sys, 'argv', test_args), \
             patch('sys.stdout', new=StringIO()) as fake_output:
            # Call the interface function which is expected to parse args and print JSON
            interface()

            # Capture the printed output
            output = fake_output.getvalue()

            # Attempt to load the output as JSON
            try:
                json_output = json.loads(output)
                # Assert that the output is a dictionary (valid JSON)
                self.assertIsInstance(json_output, dict)
            except json.JSONDecodeError:
                # If JSON decoding fails, the test should fail
                self.fail("interface() did not output valid JSON")

    #Test to ensure that the interface function outputs an error for invalid num_results.
    def test_interface_error_message(self):
        
        # Simulate command-line arguments for invalid num_results
        test_args = ["script_name", "test query", "--num_results", "16"]
        
        with patch.object(sys, 'argv', test_args), \
             patch('sys.stdout', new=StringIO()) as fake_output:
            # Call the interface function
            interface()

            # Capture the printed output
            output = fake_output.getvalue()

            # Check if the expected error message is in the output
            self.assertIn("Error: --num_results must be between 1 and 15", output)

if __name__ == "__main__":
    main()
    unittest.main()