import numpy as np
import pandas as pd
import pickle as pk
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import f1_score 
from  preprocessing import *
import time



dataset_name = "database_final.db"



#load the entire dataset
def get_dataset():
   conn = sqlite3.connect(dataset_name) 
   

   dataset = pd.read_sql_query('select * from dataset',conn)
#    dataset = pd.read_csv('dataset_2020_1.csv')   
#    #removing nan values 
#    dataset.fillna(' ',inplace = True)

   return dataset,conn



def formater(x):
    ls = []
    for i in x[1:-1].split(','):
        ls.append(re.sub('\'','',i))  
    return ls   


#calculate the f1-macro average
def score(clf,new_samples,multilabel_binarizer,vec,hard = False):

    data,conn  = get_dataset()
    new_samples = pd.DataFrame(new_samples)

    #concatenating the new samples with the dataset
    new_dataset = pd.concat([data,new_samples],axis = 0).reset_index(drop= True)
   
    #preprocess the data 
    text_data = new_dataset['Pname'] + new_dataset['Desc.']

    # get the labels
    labels = new_dataset['RC'].apply(formater)

    #pre-processing-part:
    processed_text_data = text_data.map(preprocessing)
    vectorized_text_data = vec.transform(processed_text_data)

    #binarize the labels
    labels = multilabel_binarizer.transform(labels)    

    predictions = clf.predict(vectorized_text_data)

    if hard:
        conn.execute('DROP TABLE dataset;')
        conn.commit()
        new_dataset.to_sql('dataset',conn,index = False)
        # new_dataset.to_csv(f'dataset_{time.time()}.csv',index = False)
        size = len(new_dataset)
        conn.close()
    else:
        size = len(data)    

# 
    return f1_score(predictions,labels,average = 'macro'),size

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



def retrain_soft(new_samples,clf,multilabel_binarizer,vec,hard = False):


    new_data = new_samples['Pname'] + new_samples['Desc.']
   
    


    processed_data = new_data.map(preprocessing)
    labels = new_samples['RC'].apply(formater)



    x = vec.transform(processed_data)
    y = multilabel_binarizer.transform(labels)

    # print('retrain started.')

    clf.partial_fit(x,y)

    # print('retrain done..')

    # print('please wait while scoring..')

    return score(clf,new_samples,multilabel_binarizer,vec,hard)    



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




if __name__ == '__main__':
    # testing retrain function 
    dataset,_ = get_dataset()
    new_sample = dataset.iloc[0:2]


    # print(new_sample['RC'])
    
    # score(clf,new_samples,multilabel_binarizer,vec,hard = False)

    #loading the old model 
    model = pk.load(open("model/sgdclf_final","rb"))
    mb = pk.load(open("model/multilabel_binarizer_final","rb"))
    vect = pk.load(open("model/count_vect_final","rb"))

    print(model)


    # print(retrain_soft(new_sample,model,mb,vect,hard = True))

    