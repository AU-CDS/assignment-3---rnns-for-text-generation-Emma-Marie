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

#load data
data_dir = os.path.join("in/news_data") 
# get only headlines
all_comments = []
for filename in os.listdir(data_dir):
    if 'Comments' in filename:
        article_df = pd.read_csv(data_dir + "/" + filename)
        all_comments.extend(list(article_df["commentBody"].values))
# clean a bit
all_comments = [h for h in all_comments if h != "Unknown"]
#Create corpus
corpus = [rf.clean_text(x) for x in all_comments]
# tokenization created by TensorFlow
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# turn input (every newspaper headline) into numerical output
inp_sequences = get_sequence_of_tokens(tokenizer, corpus)
# pad input (make sequences the sme length)
predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)

#create model
model = create_model(max_sequence_len, total_words)
print(model.summary())

#Train model
#history = model.fit(predictors, 
#                    label, 
#                    epochs=100, # more epochs = more accurate (the number is this low to make it run faster)
#                    batch_size=128, # large batch size to speed up the learning
#                    verbose=1)

#Save model