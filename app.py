from flask import Flask,request,jsonify
import pickle as pk

from utils import prediction


app = Flask(__name__)

@app.route('/')
def home():
    
    return 'devaition detection api working .....'

@app.route('/predict',methods = ['POST'])
def predict():

        json =  request.get_json()


        product_name = json['product_name']
        description = json['description']

        data = product_name + ' ' + description
        model = pk.load(open("model/model.pkl","rb"))
        mb = pk.load(open("model/label.pkl","rb"))
        v = pk.load(open("model/vect.pkl","rb"))
        return_data = prediction(data,model,mb,v)

    
        return jsonify(return_data)
  


if __name__ == '__main__':


    

    app.run(threaded=True, port=5000)
