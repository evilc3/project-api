from flask import Flask,request,jsonify
import pickle as pk

from utils import prediction


app = Flask(__name__)

@app.route('/')
def home():
    
    return 'devaition detection api working .....'

@app.route('/predict',methods = ['POST'])
def predict():

    model = None
    mb = None
    v = None

    try:
        data =  request.get_json()

        if type(data) != str:
            raise ValueError('Input should be string')


        model = pk.load(open("model/model.pkl","rb"))
        mb = pk.load(open("model/label.pkl","rb"))
        v = pk.load(open("model/vect.pkl","rb"))


        if model == None or mb == None or v == None:
            raise ValueError('Cannot load data...')

        return_data = prediction(data,model,mb,v)

    
        return jsonify(return_data)

    except ValueError as ve:
        
        return  str(ve)


if __name__ == '__main__':


   

    app.run()
