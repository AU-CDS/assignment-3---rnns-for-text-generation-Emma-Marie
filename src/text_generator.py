import argparse

#load RNN_model

# generates new comments

def input_parse():
    #initialie the parser
    parser = argparse.ArgumentParser()
    # add argument
    parser.add_argument("--word", type=str, default="Denmark")
    parser.add_argument("--lenght", type=int, required = True)
    # parse the arguments from command line
    args = parser.parse_args()
    #return the parsed arguments
    return args

def text(word, length):
    print (generate_text(word,length , model, max_sequence_len))

def main():
    #run input parse to get name and age
    args = input_parse()
    # print generated text
    text(args.word, args.length)

if __name__=="__main__":
    main()