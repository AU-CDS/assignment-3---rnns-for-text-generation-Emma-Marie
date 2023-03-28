# data processing tools
import string, os 
import pandas as pd
import numpy as np
np.random.seed(42)
# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(42)
import tensorflow.keras.utils as ku 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
# utils
import sys
sys.path.append(".")
import utils.requirement_functions as rf

def get_data():
    #load data
    #data_dir = os.path.join("in/news_data") 
    data_dir = os.path.join("in/test_data") 
    # get only headlines
    all_comments = []
    for filename in os.listdir(data_dir):
        if 'Comments' in filename:
            article_df = pd.read_csv(data_dir + "/" + filename)
            all_comments.extend(list(article_df["commentBody"].values))

    
    #sampling 10000 random comments
    import random
    sample_comments = random.sample(all_comments, 10000)
    
    # clean a bit
    # OBS remember to change sample_comments into all_comments
    sample_comments = [h for h in sample_comments if h != "Unknown"]
    #Create corpus
    corpus = [rf.clean_text(x) for x in sample_comments]
    
    # tokenization created by TensorFlow
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    # turn input (every newspaper headline) into numerical output
    inp_sequences = rf.get_sequence_of_tokens(tokenizer, corpus)
    # pad input (make sequences the same length)
    predictors, label, max_sequence_len = rf.generate_padded_sequences(inp_sequences, total_words)

    return max_sequence_len, total_words, predictors, label


def rnn_model(max_sequence_len, total_words, predictors, label):
   #create model
    model = rf.create_model(max_sequence_len, total_words)
    print(model.summary())
    #Train model
    history = model.fit(predictors, 
                        label, 
                        epochs=10, #turn epoc up
                        batch_size=128,
                        verbose=1)

    return model

def main():
   # load and prepare data
   max_sequence_len, total_words, predictors, label = get_data()
   print("Data is prepared")
   # train model
   model = rnn_model(max_sequence_len, total_words, predictors, label)
   print("Model is trained!")
   # Save model
   outpath = os.path.join("out/rnn_model")
   model.save(outpath, overwrite=True, save_format=None)
   #tf.keras.saving.save_model(model, outpath, overwrite=True, save_format=None)
   print("Model saved!")

if __name__ == "__main__":
    main()