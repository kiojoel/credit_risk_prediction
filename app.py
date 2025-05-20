import joblib
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
model=joblib.load('credit_risk_model.pkl')
scaler=joblib.load('scaler.pkl')

@app.route('/')
def home():
  return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
  data=request.json['data']
  print(data)
  print(np.array(list(data.values())).reshape(1,-1))
  new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
  prediction=model.predict(new_data)
  output=prediction[0]
  if output==1:
    return jsonify({'prediction':'The person is likely to default on the loan'})
  else:
    return jsonify({'prediction':'The person is not likely to default on the loan'})

if __name__=="__main__":
  app.run(debug=True)