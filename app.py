from flask import Flask,request,jsonify
import pickle as pk
import datetime
from utils import *


model_name = 'sgdclf_final'
label_name = 'multilabel_binarizer_final'
vect_name = 'count_vect_final'


app = Flask(__name__)

@app.route('/')
def home():
    
    return 'deviation detection api working .....'

@app.route('/predict',methods = ['POST'])
def predict():

    model = None
    mb = None
    v = None

    try:
        data =  request.get_json()

        if type(data) != str:
            raise ValueError('Input should be string')


        model = pk.load(open(f"model/{model_name}","rb"))
        mb = pk.load(open(f"model/{label_name}","rb"))
        v = pk.load(open(f"model/{vect_name}","rb"))


        if model == None or mb == None or v == None:
            raise ValueError('Cannot load data...')

        return_data = prediction(data,model,mb,v)

    
        return jsonify(return_data)

    except ValueError as ve:
        
        return  str(ve)


@app.route('/retrain/<int:hard>',methods = ['POST'])
def retrain(hard):
    
    '''
    THINGS TO DO IN THIS METHOD:
    1> GET THE DATA FROM THE USER*
    2> PRE-PROCESS THE DATA* 
    3> USE PARTIAL-FIT TO RE-TRAIN THE MODEL ON THE DATA*
    4> APPEND THE DATA TO DATASET.CSV FILE 
    5> SENT INFO MESSAGE TO THE USER 
        THE INFO MESSAGE SHOULD CONTAIN 
        1. THE UPDATED MODEL SCORE 
        2. THE UPDATED SIZE OF THE DATASET
        3. TIME OF RETRAIN 
        4. THAT RETRAIN WAS SUCESSFUL.
    6> Exception Handling
    '''
    
    model = None
    mb = None
    v = None

    try:
        data =  request.get_json()

        data  = pd.DataFrame(data)

        print(data)    

        #loading the old model 
        model = pk.load(open(f"model/{model_name}","rb"))
        mb = pk.load(open(f"model/{label_name}","rb"))
        v = pk.load(open(f"model/{vect_name}","rb"))


        if model == None or mb == None or v == None:
            raise ValueError('Cannot load data...')


        score,size =  retrain_soft(data,model,mb,v,hard)

        time  =   datetime.datetime.today().strftime("%c")

        return_message = {'score':score,'status':'sucessfull','dataset_size':size,'time': time}


        return jsonify(return_message)

    except ValueError as ve:
        return str(ve) 





if __name__ == '__main__':

    app.run()
