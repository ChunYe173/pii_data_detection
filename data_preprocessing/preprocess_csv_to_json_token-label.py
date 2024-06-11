import os
import json
import pandas as pd
import re

# Output a file that allow break down of the labelling of all the tokens
# Validate tokens in dataset are labelled correctly
if __name__ == "__main__":
    csv_file = input("Please input filename of csv to process (i.e. filename.csv): ")
    print(csv_file)
    output_filename = input("Please input filename of output json file (i.e. filename.txt): ")
    print(output_filename)

    # csv_file = 'wnut17train_cleaned.csv' 
    # output_filename = "wnut17train_cleaned_raw.txt"

    output_file = open(output_filename, 'w', encoding='utf-8')
    df = pd.read_csv(csv_file)
    
    print("Loaded files.")

    for i in df.index:
        print("Index: ", i)
        # print(df['tokens'][i])
        # print(df['labels'][i])
        # print("Length token list: ", len(eval(df['tokens'][i])))
        # print("Length label list: ", len(eval(df['labels'][i])))
        # print("Length token type: ", type(df['tokens'][i]))
        # print("Length label type: ", type(df['labels'][i]))
        assert (len(eval(df['tokens'][i])) == len(eval(df['labels'][i])))
        for j in range(len(eval(df['tokens'][i]))):
            a = eval(df['tokens'][i])[j]
            # if '\n' in a:
            #     a = r"{}".format(a)
            b = eval(df['labels'][i])[j]
            # output_file.write('\%s\t%s' %(eval(df['tokens'][i])[j], eval(df['labels'][i])[j]))
            output_file.write('%s\t%s' %(a, b))
            output_file.write('\n')
        output_file.write('\n')

    output_file.close()

    print("Generated text file with break down of token-label from dataset csv.")

