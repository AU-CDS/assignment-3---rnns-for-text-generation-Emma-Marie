[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-8d59dc4de5201274e310e4c54b9627a8934c3b88527886e3b421487c677d23eb.svg)](https://classroom.github.com/a/5f7lMH9Y)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10586695&assignment_repo_type=AssignmentRepo)
# Assignment 3 - Language modelling and text generation using RNNs

## Purpose
In this assignment, I will make use of TensorFlow so build a recurrent neural network model for NLP. The model is trained on the ```New York Times comments```  data set. The model can be used to generate new newspaper comments based on predictions of which word are most likely to follow the former word.

## Scripts
This project consists of two scripts, which can both be found in the ```src```folder: 
    1) The first script ```RNN_model.py``` loads and prepares the data, and train and save the model. The preparetion consists of creating a corpus of only the csv files with comments (and not articles), tokenizing the texts, turn the texts into numerical output, and pad the outputs to give them the same length.  The trained model is saved in the ```out``` folder as well as the tokenizer. The model is mostly a proof of concept (see 'Notes on the model'). The name of the model includes the numeric value of the max_sequence_len variable, which is needed in the script text_generator.py. This value is isolated in text_generator.py using regex, and then assigned to a max_sequence_len in this script. 

    2) The second script ```text_generator.py``` loads in the model and uses it to generate new comments. The generated comment is based on two arguments parsed through the commandline. The arguments are the begining word and the length of the generated text (number of words after beginning word). 

## Data
The data used to train the model is the ```New York Times comments```  data set. It consists of information about the comments, which have been made on New York Times articles in the period January-May 2017 and Januari-April 2018. 

Before running the script, you must download the data set through Kaggle and place it in the ```in``` folder and call it "news_data". The data can be downloaded (503 MB) here: https://www.kaggle.com/datasets/aashita/nyt-comments?select=ArticlesApril2017.cs

Note that the data consists of two csv files, one for each article and one for the comments made on the given article. The data consists of over 2 million comments with 34 features.

__NB:__ make sure not to push the data to GitHub, because of the size of the data files.

## How to run the scripts
- Download the data set from Kaggle and put it in the ```in``` folder. Call the data "news_data" or remember to change the file path, if you give it another name. 
- Begin by running the RNN_model.py:
    - run "bash setup.sh" from the commandline to create a virtual environment and install the required packages.
    - run "bash run.sh" from the commandline to activate the virtual environment, run the code, and deactivate the environment. 
- Then run the text_generator.py:
    - The text_generator.py is run through the commandline by typing "python3 src/text_generator.py --filename ```write filename```-- start_word ```write word```--length ```write an int number```". The filename is the name of the model trained in the RNN_model.py script. The start_word a self chosen word which should be the first word of the generated text. The length is a number indicating how many words you wish to follow your chosen word in the generated text. 

## Notes on the model
The data set is very large, and I had to train my model on a smaller sample to be able to train it without ucloud killing the run. I created a repo called test_data consisting of two of the csv's and did a random sample picking 100 comments.  
I also only set the epochs in the model to 100, to make the model train faster. The epochs could be given a bigger value for the model to make more accurate predictions. My model is therefore rather a __proof of concept__ than an actually well performing model.

## Utils
In the ```utils``` folder there is a script called ```requirement_functions.py```, which are imported into my two main scripts. The functions in this requirement_functions.py script is developed in class.

## References
"New York Times Comments", New York Times, Kaggle (how to reference properly?????)