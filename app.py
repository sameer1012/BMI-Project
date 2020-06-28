import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
   
    output = round(prediction[0], 2)
    if int(output)==0:
        output1='Extremely Weak'
    elif int(output)==1:
        output1='Weak'
    elif int(output)==2:
        output1='Healthy'
    elif int(output)==3:
        output1='Over Weight'
    elif int(output)==4:
        output1='Obese'
    elif int(output)==5:
        output1='Extremely Obese'
    return render_template('index.html', prediction_text='Person is in  {}'.format(output1)+' condition')



if __name__ == "__main__":
    app.run(debug=True)