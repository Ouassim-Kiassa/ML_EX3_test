import pandas as pd
from sklearn.datasets import load_files
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def to_tabular(path):
    df = load_files(f"{path}")

    df = pd.DataFrame([df.data, df.target.tolist()]).T
    df.columns = ['text', 'target']
    
    return df

def twenty_newsgroup_to_csv():
    newsgroups = load_files(r"./datasets/20_newsgroups")

    df = pd.DataFrame([newsgroups.data, newsgroups.target.tolist()]).T
    df.columns = ['text', 'target']
    df.to_csv('20_newsgroup.csv')


def pos_neg_to_csv():
    pos_neg = load_files(r"./datasets/aclImdb/train")

    df = pd.DataFrame([pos_neg.data, pos_neg.target.tolist()]).T
    df.columns = ['text', 'target']
    df.to_csv('pos_neg.csv')
    
def scc(df):
    df['Content_Parsed_1'] = df['text'].str.replace("\r", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("\n", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("    ", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace('"', '')
    
    return df

def lowercase(df):
    df['Content_Parsed_2'] = df['Content_Parsed_1'].str.lower()
    
    return df

def punc(df):
    punctuation_signs = list("?:!.,;")
    df['Content_Parsed_3'] = df['Content_Parsed_2']

    for punct_sign in punctuation_signs:
        df['Content_Parsed_3'] = df['Content_Parsed_3'].str.replace(punct_sign, '')
        
    return df

def posses(df):
    df['Content_Parsed_4'] = df['Content_Parsed_3'].str.replace("'s", "")
    
    return df

def lem(df):
    wordnet_lemmatizer = WordNetLemmatizer()

    nrows = len(df)
    lemmatized_text_list = []

    for row in range(0, nrows):

        # Create an empty list containing lemmatized words
        lemmatized_list = []

        # Save the text and its words into an object
        text = df.loc[row]['Content_Parsed_4']
        text_words = text.split(" ")

        # Iterate through every word to lemmatize
        for word in text_words:
            lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

        # Join the list
        lemmatized_text = " ".join(lemmatized_list)

        # Append to the list containing the texts
        lemmatized_text_list.append(lemmatized_text)

    df['Content_Parsed_5'] = lemmatized_text_list
        
    return df

def stopword_removal(df):
    stop_words = list(stopwords.words('english'))

    df['Content_Parsed_6'] = df['Content_Parsed_5']

    for stop_word in stop_words:

        regex_stopword = r"\b" + stop_word + r"\b"
        df['Content_Parsed_6'] = df['Content_Parsed_6'].str.replace(regex_stopword, '')

    return df

def final(df):
    list_columns = ["text", "Content_Parsed_6"]
    df = df[list_columns]

    df = df.rename(columns={'Content_Parsed_6': 'text_parsed'})
    
    return df