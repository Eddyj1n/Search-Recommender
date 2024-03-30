# Search Recommender

This project creates a simple search recommender system for user searches, providing recommendations on related topics to explore based on an initial search query.

## Context

This mini project was developed to gain exposure to Natural Language Processing (NLP) principles, given the recent hype in Generative AI and Transformers. It covers rudimentary principles such as tokenization, pre-processing of text, similarity measures, and word embeddings.

Many websites host millions of individual pieces of content, covering a wide range of topics. To help users explore the available content effectively, providing recommendations on related topics can greatly enhance their experience.

## Data

The data files for this project are in delimited (CSV, TSV) format:

- Content tags: CSV file containing tags assigned to content items (~100,000 rows)
- User search keywords: TSV file containing keywords people searched for in a visit (~921,773 rows)

## Objective

The objective of this project is to build a working system in Python that recommends topics related to a given keyword or phrase. By leveraging the provided data, the system aims to assist users in discovering relevant content based on their search queries.

## Features

- Tokenization and pre-processing of text data
- Calculation of similarity measures between search queries and content tags
- Utilization of word embeddings to capture semantic relationships
- Generation of topic recommendations based on user searches

## Installation

1. Clone the repository:
