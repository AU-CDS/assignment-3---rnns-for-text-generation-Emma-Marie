import argparse
import tensorflow as tf
import tensorflow.keras.utils as ku 
from keras.models import load_model
from tensorflow import keras
import sys
sys.path.append(".")
import utils.requirement_functions as rf
import RNN_model

def main():
    #load RNN_model
    model = keras.models.load_model('out/rnn_model')
    #model = load_model('out/rnn_model')
    #tf.keras.saving.load_model(filepath, custom_objects=None, compile=True, safe_mode=True)
    max_sequence_len = RNN_model
    tokenizer = RNN_model
    print (rf.generate_text("danish", 5, model, max_sequence_len))

#def input_parse():
#    #initialie the parser
#    parser = argparse.ArgumentParser()
#    # add argument
#    parser.add_argument("--word", type=str, default="Denmark")
#    parser.add_argument("--lenght", type=int, required = True)
#    # parse the arguments from command line
#    args = parser.parse_args()
#    #return the parsed arguments
#    return args

#def text(word, length):
#    print (generate_text(word, length, model, max_sequence_len))

#def main():
#    #run input parse to get name and age
#    args = input_parse()
#    # print generated text
#    text(args.word, args.length)

if __name__ == "__main__":
    main()