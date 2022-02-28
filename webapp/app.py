from flask import Flask, render_template, request, jsonify
import flask 
import numpy as np
import traceback
import pickle
import pandas as pd
 
 
# App definition
app = Flask(__name__)
 
# importing models
with open('model/elastic_net.pkl', 'rb') as f:
   classifier = pickle.load (f)
 
with open('model/model_columns.pkl', 'rb') as f:
   model_columns = pickle.load (f)
with open('model/sum_columns.pkl', 'rb') as f:
   sum_columns = pickle.load (f)
 
 
@app.route('/')
def welcome():
   return render_template('main.html')
    # return "hello"
 
@app.route('/predict', methods=['POST'])
def predict():
   if flask.request.method == 'POST':
       try:
           json_ = request.form.to_dict()
    
           query_ = pd.get_dummies(pd.DataFrame(json_,  index=[0]), columns=model_columns)
           query = query_.reindex(columns = sum_columns, fill_value= 0)
           prediction = classifier.predict(query)
           prediction= prediction*1000000000
        #    return jsonify({
        #        "prediction":str(prediction)
        #    })
           return render_template('main.html', prediction_text='Giá dự đoán: {}'.format(prediction))
 
       except:
           return jsonify({
               "trace": traceback.format_exc()
               })
        # to_predict_list = request.form.to_dict()
        # return  to_predict_list
 
if __name__ == "__main__":
   app.run(debug=True)

