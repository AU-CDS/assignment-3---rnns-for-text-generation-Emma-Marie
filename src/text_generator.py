import argparse
import sys
sys.path.append(".")
import utils.requirement_functions as rf
import tensorflow as tf
from keras.models import load_model
from tensorflow import keras
# loading tokenizer
from joblib import load
def input_parse():
    #initialie the parser
    parser = argparse.ArgumentParser()
    # add argument
    parser.add_argument("--filename", type=str) # name of the model
    parser.add_argument("--start_word", type=str) #the first word of the generated text
    parser.add_argument("--length", type=int, default=10) # the number of words following the chosen word
    # parse the arguments from command line
    args = parser.parse_args()
    return args
def load_model(args):
    # set filename 
    filename = args.filename
    # importing trained RNN model
    model = tf.keras.models.load_model(f"model/{filename}")
    # load tokenizer
    tokenizer = load("model/tokenizer.joblib")
    # get max_sequence_len from filename
    max_sequence_len = filename.split("_")[1].split("q")[1].split(".")[0]
    return model, max_sequence_len, tokenizer
def main():
    args = input_parse()
    model, max_sequence_len, tokenizer = load_model(args)
    #print model summary
    model.summary()
    #generate new text
    print(rf.generate_text(tokenizer, args.start_word, args.length, model, max_sequence_len))
if __name__ == "__main__":
    main()