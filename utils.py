import numpy as np
import pickle as pk
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import re
from string import punctuation


from list_utils import *


word_punct = set(words).union(punctuation).union(extended)

#decode the root cause and the category
def decode_cat(ops):
    
    cat_ops = []    
    for k in ops:
        cat = [] 
        for y in k:
        
            if y.strip() == 'unknown':
                cat.append(y)
                break
        
            for i,j in enumerate(list_rc_sub):
            
                if y.strip() in j:
                    cat.append(list_rc_cat[i])
                    break
                
        cat_ops.append(cat)
    return cat_ops


#filters out punctuations, special characters ,numbers (optional), stopwords
def preprocessing(x):
    
    input = re.sub('[!@#$%^&*()\n_:><?\-.{}|+-,;""``~`—]|[0-9]|/|=|\[\]|\[\[\]\]',' ',x)
    input = re.sub('[“’\']','',input)  
    tmp = " "
    
    for i in input.split(" "):
        
        if i not in word_punct:
            tmp += i.lower() + " ";
    
    return tmp    


#api.py will be calling only this fuction...
def prediction(x,clf,multilabel_binarizer,vec):

    processed_text = preprocessing(x)

    data = vec.transform([processed_text])

    ops = clf.predict(data)
    
    labels = multilabel_binarizer.inverse_transform(ops)



    ops_prob = clf.predict_proba(data) * ops

    labels_prob = multilabel_binarizer.classes_[ops_prob[0]>0]

    ops_list = ops_prob[ops_prob != 0]
    


    cat = decode_cat(labels)


    '''
    
    return data -> {categories:[...],root_cause:[.....],proba:[......]}

    '''




    if len(cat[0]) > 0:
        
        
        categories = ""
        for i in cat[0][:]:
            categories += i + ","

        ops_list *= 100
        data = {'cat':[categories],'root_causes':labels,'proba':ops_list.tolist()}

        return data
        
        # for i,j in enumerate(labels[0]):
        #     st.write("### "+j.strip() + 'proba'ops_list[i] * 100)
    else:
        return "no root cause deteced please enter valid input" 

