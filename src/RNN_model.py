# data processing tools
import string, os 
import pandas as pd
import numpy as np
np.random.seed(42)
# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.preprocessing.text import Tokenizer

# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
# utils
import sys
sys.path.append(".")
import utils.requirement_functions as rf

# set path to news data
data_dir = os.path.join("in", "news_data") 

def get_data(path):
    # get only the comments from the data
    all_comments = []
    # go over each file in the dir
    for filename in os.listdir(path):
        # check if "Comments" are part of the file name
        if 'Comments' in filename:
            # if so, then ass the filename to a df
            comments_df = pd.read_csv(path + "/" + filename)
            # add the column with the content of the comments to the all_comments list
            all_comments.extend(list(comments_df["commentBody"].values))

    # clean comments a bit
    all_comments = [h for h in all_comments if h != "Unknown"]
    #create corpus
    corpus = [rf.clean_text(x) for x in all_comments]
    
    # tokenization created by TensorFlow
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    # turn the comments into numerical output
    inp_sequences = rf.get_sequence_of_tokens(tokenizer, corpus)
    # pad (make sequences the same length)
    predictors, label, max_sequence_len = rf.generate_padded_sequences(inp_sequences, total_words)
    print("Data is prepared")

    return max_sequence_len, total_words, predictors, label, tokenizer


def rnn_model(max_sequence_len, total_words, predictors, label, outpath):
   #create model
    model = rf.create_model(max_sequence_len, total_words)
    print(model.summary())
    #Train model
    history = model.fit(predictors, 
                        label, 
                        epochs=1, # number of epochs
                        batch_size=128, # the size of batches of data
                        verbose=1)
    # save the model
    tf.keras.models.save_model(model, outpath, overwrite=False, save_format=None)
    print("Model saved!")

    return model

def main():
   # load and prepare data
   max_sequence_len, total_words, predictors, label, tokenizer = get_data(data_dir)
   # train model
   model_path = os.path.join(f"model", f"rnn-model_seq{max_sequence_len}.keras")
   model = rnn_model(max_sequence_len, total_words, predictors, label, model_path)
   # saving tokenizer
   from joblib import dump, load
   tokenizer_path = os.path.join("model", "tokenizer.joblib")
   dump(tokenizer, tokenizer_path)
   
if __name__ == "__main__":
    main()