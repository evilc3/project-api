import re
from string import punctuation
from list_utils import *
from nltk.stem import PorterStemmer



stem = PorterStemmer()


word_punct = set(words).union(punctuation).union(extended)

#filters out punctuations, special characters ,numbers (optional), stopwords

def preprocessing(x):
    
    input = re.sub('[!@#$%^&*()\n_:><?\-.{}|+-,;""``~`—]|/|=|\[\]|\[\[\]\]',' ',x)
    input = re.sub('[“’\']','',input)  
    tmp = " "
    
    for i in input.split():
        
        if i not in word_punct:
            tmp += stem.stem(i.lower()) + " ";
    
    return tmp    
    