[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-8d59dc4de5201274e310e4c54b9627a8934c3b88527886e3b421487c677d23eb.svg)](https://classroom.github.com/a/5f7lMH9Y)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10586695&assignment_repo_type=AssignmentRepo)
# Assignment 3 - Language modelling and text generation using RNNs

## Purpose
In this assignment, I will make use of TensorFlow so build a recurrent neural network model for NLP. The model is trained on the ```New York Times comments```  data set. The model can be used to generate new newspaper comments based on the ...

The trained model is saved in the ```out``` folder. The model is mostly a proof of concept (see 'Notes on the model')
All scripts can be found in ```src```folder. 

## Scripts
This project consists of two scripts: 
    1) the first script ```RNN_model.py``` loads and prepares the data, and train and save the model. The preparetion consists of creating a corpus of only the csv files with comments (and not articles), tokenizing the texts, turn the texts into numerical output, and pad the outputs to give them the same length.  
    2) The second script ```text_generator.py``` loads in the model and uses it to generate new comments. The generated comment is based on two arguments parsed through the commandline. The arguments are the begining word and the length of the generated text (number of words after beginning word). 

## Data
The data used to train the model is the ```New York Times comments```  data set. It consists of information about the comments, which have been made on New York Times articles in the period January-May 2017 and Januari-April 2018. 

Before running the script, you must download the data set through Kaggle and place it in the ```in``` folder and call it "news_data". The data can be downloaded (503 MB) here: https://www.kaggle.com/datasets/aashita/nyt-comments?select=ArticlesApril2017.cs

Note that the data consists of two csv files, one for each article and one for the comments made on the given article. The data consists of over 2 million comments with 34 features.

NB: make sure not to push the data to GitHub, because of the size of the data files.

## Notes on the model
The data set is very large, and I had to train my model on a smaller sample to be able to train it without ucloud killing the run. I created a repo called test_data consisting of two of the csv's and did a random sample picking 1000 comments.  
I also only set the epochs in the model to 10, to make the model train faster. The epochs could be given a bigger value for the model to make more accurate predictions. Even with the small data sample and the few epocs it took me 7 hours to train the model. My model is therefore rather a __proof of concept__ than an actually well performing model.

## How to run the scripts
- download the data set from Kaggle and put it in the ```in``` folder. Call the data "news_data" or remember to change the file path, if you give it another name. 
- run "bash setup.sh" from the commandline to create a virtual environment and install the required packages.
- run "bash run.sh" from the commandline to activate the virtual environment, run the code, and deactivate the environment. 

__INSTRUCTION TO GENERATE NEW TEXT! (HOW TO PARSE AN WORD AND AN LENGTH ARGUMENT)__

## References