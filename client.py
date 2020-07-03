import requests
import json


def call(url,data):
      
    

      j_data = json.dumps(data)



      headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

      r = requests.post(url, data=j_data, headers=headers)
      print(r.text)


# response = json.loads(r.text)

# print(response['cat'],)

      


# predict route.
      
# url = 'https://deviationdetectorapi.herokuapp.com/predict'

# data = 'Methocarbamol tablets 750mg. the product was achieved out of limit against the limit mentioned in BMR. observed value is 1.18% and mentioned BMR is 1%.'
        
# call(url,data)


#retrain route 


# url = 'https://deviationdetectorapi.herokuapp.com/retrain'


url = 'http://127.0.0.1:5000/retrain/0'

data  = { 
          'Pname':['Methocarbamol tablets 750mg.'],
          'Desc.':['The product was achieved out of limit against the limit mentioned in BMR.  observed value is 1.18% and mentioned BMR is 1%.'],
          'RC':["['Procedure', ' Analysis', ' batchsize', ' environment']"],
          'CA':['calibration must be required to change range of temperature. collection of samples required.']  
        }




call(url,data)



