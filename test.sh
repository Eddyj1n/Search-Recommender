#!/bin/bash

# This script tests your application for recommended terms
if [ -f requirements.txt ]; then 
    pip install -r requirements.txt
fi

# run the intialisation
python init.py

# generate some recommendations

# should return valid values
python recommend.py "hot dog"
python recommend.py --num_results 5 "travel"
python recommend.py --num_results 1 "butterfly"

# should return appropriate error messages
python recommend.py --num_results 25 "hot dog"
python recommend.py --num_results 0
python recommend.py --num_results "hello"
