import argparse

import tensorflow
import tensorflow.keras.utils as ku 
from keras.models import load_model
from tensorflow import keras

import sys
sys.path.append(".")
import utils.requirement_functions as rf


def main():
    #load RNN_model
    model = keras.models.load_model('out')
    #model = load_model('model.h5')
    #from RNN_model.py import max_sequence_len
    print (rf.generate_text("danish", 5, model, max_sequence_len))
# generates new comments
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
#    print (generate_text(word,length , model, max_sequence_len))

#def main():
#    #run input parse to get name and age
#    args = input_parse()
#    # print generated text
#    text(args.word, args.length)

if __name__ == "__main__":
    main()