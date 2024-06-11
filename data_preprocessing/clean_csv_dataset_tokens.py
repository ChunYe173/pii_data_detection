import os
import json
import pandas as pd

 ### Note that this code will only work at token level and not on the full_text - 2024 March 10

if __name__ == "__main__":
    csv_file = input("Please input filename of csv to process (i.e. filename.csv): ") 
    print(csv_file)
    output_filename = input("Please input filename of output csv file (i.e. filename.csv): ")
    print(output_filename)

    # csv_file = 'train.csv'
    # output_filename = "train_cleaned.csv"

    print('Loading csv.')
    df = pd.read_csv(csv_file)
    for col in ['tokens','trailing_whitespace','labels']:
        df[col] = df[col].apply(lambda x: eval(x))
    print('Sample data.')
    print(df.head(3))

    # Populate list with specific characters to remove from string in token
    char_to_remove = [
    '''   
        
      ''',
    '''   
     ''',
     '​—​','​',' ','​“','—','–','‐','â€œ‹','Â­â€',
     '­','\u00c9','\u201c','\u201d','â€¢','\u2022','â€¦','â€ ','â€“','Ã‰','Â©','™','Â','â','€','œ',
     '‹','“','”','â€™','\u200b','’','‘','©','…','�','ï¿½']
     
     # if tokens are as such, remove from training data
    remove_token_from_list = ['']

    # Cleaning all tokens. Omitted from cleaning: full_text. Can edit code to clean full_text too. 
    # Loop through every single element in all token list
    for i in df.index:
        print("Processing row: ", i)
        indexes_to_remove = []
        for j in range(len(df['tokens'].iloc[i])):
            for char in char_to_remove:
                if char in df['tokens'].iloc[i][j]:
                    df['tokens'].iloc[i][j] = df['tokens'].iloc[i][j].replace(char,'')
            if 'ï¬‚' in df['tokens'].iloc[i][j]: # 'ﬂ'
                df['tokens'].iloc[i][j] = df['tokens'].iloc[i][j].replace('ï¬‚','fl')
            # To mark out tokens for removal from the list
            if df['tokens'].iloc[i][j] in remove_token_from_list:
                indexes_to_remove.append(j)
        indexes_to_remove.reverse()
        # remove tokens from dataset
        for index in indexes_to_remove:
            df['tokens'].iloc[i].pop(index)
            df['trailing_whitespace'].iloc[i].pop(index)
            df['labels'].iloc[i].pop(index)
        assert len(df['tokens'].iloc[i]) == len(df['labels'].iloc[i])
        assert len(df['tokens'].iloc[i]) == len(df['trailing_whitespace'].iloc[i])
    df.to_csv(output_filename, index=False)

    print("Generated cleaned csv file.")

"""
#trash strings: ​,  , â€œ‹, \n\n, ' ', '    ', '   ', \n,­,™,Â­â€,
Â­â€,\xa0,\u2022,

    
    
   ,

  ,
 Â,

    
   ,

Document 1339
NAME
Ruben Pabon,
Georgia,
Bethany,
Xiaoxuan Yang,
Fernando    O
Castro  O
,
    O
1960    O
,   O
pp  O
.   O
1–44    O
.   O
,
Rice    O
University  O
’s  O
Baker   O
Institute   O

Trash strings that cannot be processed: â€¢, 
"""