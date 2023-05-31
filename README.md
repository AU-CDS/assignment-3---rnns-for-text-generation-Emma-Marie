[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-8d59dc4de5201274e310e4c54b9627a8934c3b88527886e3b421487c677d23eb.svg)](https://classroom.github.com/a/5f7lMH9Y)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10586695&assignment_repo_type=AssignmentRepo)
# Assignment 3 - Language modelling and text generation using RNNs

## 1. Contribution 
I have developed the code for the two main scripts for this assignment without other contributors. In the two main scripts I use some functions from the script ```required_functions.py``` in the ```utils``` folder. These functions were provided by my teacher Ross and used during class.

## 2. Description
In this assignment, I will use ```TensorFlow``` to build a recurrent neural network model for NLP. The model is trained on the ```New York Times comments``` dataset. The model can be used to generate new newspaper comments based on predictions of which word are most likely to follow the former. 

## 3. Methods
This assignment consists of two scripts, which can be found in the ```src``` folder. The first script is ```RNN_model.py```. It loads and prepares the data and trains and saves a RNN model. The preparation consists of creating a corpus of the csv files with comments, tokenizing the comments, turning them into numerical output, and pad the outputs to give them the same length.  Then the RNN model is set up and trained on the dataset, and the trained model is saved in the models folder as well as the tokenizer. The name of the saved model includes the numeric values of the max_sequence_len variable, which is needed in the script text_generator.py. 

The second script is called ```text_generator.py```. The script loads the trained model and uses it to generate new newspaper comments. The value of the variable max_sequence_len is extracted from the name of the model using regex. The generated comment is based on the three arguments parsed through the command line. The arguments are the beginning word and the length of the generated comment.  

## 4. Data
The data used for training the model is the ```New York Times comments``` dataset. It consists of information about the comments, which have been made on New York Times articles in the period January-May 2017 and January-April 2018. The data consists of over 2 million comments with 34 features. Note that the dataset consists of 18 csv files, 9 with articles and 9 with comments. For this assignment only the csvs with comments are used. 

### 4.1 Get the data
Before running the script, please download the dataset (503 MB) through Kaggle: https://www.kaggle.com/datasets/aashita/nyt-comments?select=ArticlesApril2017.cs. Name the dataset “news_data” and place it in the ```in``` folder. 

## 5. Usage
### 5.1 Prerequisites
For the scripts to run properly, please install Python 3 and Bash. The code for this assignment is created and tested using the app “Coder Python 1.73.1” on Ucloud.sdu.dk. The final step it to clone the GitHub repository on your own device.

### 5.2 Install packages
To install the required packages, run the command “bash setup.sh” from the command line. The command creates a virtual environment and installs the packages from requirements.txt in it:

        bash setup.sh


### 5.3 Run the scripts
The script ```RNN_model.py``` is run with the comment “bash run.sh” from the command line. The command activates the virtual environment, run the script, and deactivates the environment: 
		
		bash run.sh

To run the script ```text_generator.py```, two steps must be followed. First, you have to manually activate the virtual environment by running the command “source ./tensorflow_env/bin/activate” from the command line:

        source ./tensorflow_env/bin/activate

Then, run the command “python3 src/text_generator.py --filename --start_word --length”. The filename argument is the name of the model trained in the RNN_model.py. This argument is necessary, because the max_sequence_len is extracted from the model’s name in the generator script. The start_word argument is the word which you wish to be the first word of the generated piece of text. The length argument is the number of words you wish to follow your chosen word in the generated piece of text. If no number is specified,  the default length is 10 words, and 10 us also used in the example below:

        python3 src/text_generator.py --filename rnn-model_seq280.keras --start_word female --length 10

## 6. Descussion of results
The dataset is very large, and I had to train my model on a smaller sample to be able to train it without Ucloud killing the run. The model is only trained on 100 sampled comments, and it is therefore rather a proof of concept than an actual well performing model. This means, that the generated text isn’t as convincing as it would be, if the model were trained on a larger data sample. 

## 7. References
"New York Times Comments", New York Times, Kaggle: https://www.kaggle.com/datasets/aashita/nyt-comments 
