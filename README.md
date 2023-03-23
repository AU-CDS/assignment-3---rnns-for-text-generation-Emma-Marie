[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-8d59dc4de5201274e310e4c54b9627a8934c3b88527886e3b421487c677d23eb.svg)](https://classroom.github.com/a/5f7lMH9Y)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10586695&assignment_repo_type=AssignmentRepo)
# Assignment 3 - Language modelling and text generation using RNNs

## Purpose
In this assignment, I will make use of TensorFlow so build a model for NLP. 

Comment on this: It is mostly a proof of concept (model don't perform that well).

Model can be found in ```out```folder. 
Scripts can be found in ```src```folder. 
Other stuff: setup.sh, run.sh, requirements.txt, utils (requirement_functions.py)

## Scripts


## Data
The data used to train the model is the ```New York Times comments```  data set. It consists of information about the comments, which have been made on New York Times articles in the period January-May 2017 and Januari-April 2018. 

Before running the script, you must download the data set through Kaggle and place it in the ```in``` folder and call it "news_data". The data can be downloaded (503 MB) here: https://www.kaggle.com/datasets/aashita/nyt-comments?select=ArticlesApril2017.cs

Note that the data consists of two csv files, one for each article and one for the comments made on the given article. The data consists of over 2 million comments with 34 features.

NB: make sure not to push the data to GitHub, because the data files are too big. 

## How to run the scripts
- download the data set from Kaggle and put it in the in folder.
- run "bash setup.sh" from the commandline to create a virtual environment and install the required packages
- run "bash run.sh" from the commandline to activate the virtual environment, run the code, and deactivate the environment. 

INSTRUCTION TO GENERATE NEW TEXT! (HOW TO PARSE AN WORD AND AN LENGTH ARGUMENT)

## References