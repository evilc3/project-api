from flask import Flask,request,jsonify
import pickle as pk

from utils import prediction


app = Flask(__name__)

@app.route('/')
def home():
    
    return 'hello world'

@app.route('/predict',methods = ['POST'])
def predict():

    try:
        json =  request.get_json()


        product_name = json['product_name']
        description = json['description']

        data = product_name + ' ' + description

        return_data = prediction(data,model,mb,v)

    
        return return_data
    except Exception:
        return 'some error has occured' + Exception.__str__


if __name__ == '__main__':


    model = pk.load(open("../model/model.pkl","rb"))
    mb = pk.load(open("../model/label.pkl","rb"))
    v = pk.load(open("../model/vect.pkl","rb"))

    app.run(threaded=True, port=5000)
