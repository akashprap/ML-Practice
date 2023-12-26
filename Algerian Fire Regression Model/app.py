from flask import Flask,request,render_template,jsonify
from sklearn.preprocessing import StandardScaler
import pickle

app=Flask(__name__)
ridge_model=pickle.load(open('model/ridge.pkl','rb'))
scaler_model=pickle.load(open('model/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=="POST":
        temperature = float(request.form.get('temperature'))
        rh = float(request.form.get('rh'))
        ws = float(request.form.get('ws'))
        rain = float(request.form.get('rain'))
        ffmc = float(request.form.get('ffmc'))
        dmc = float(request.form.get('dmc'))
        isi = float(request.form.get('isi'))
        classes = float(request.form.get('classes'))
        region = float(request.form.get('region'))
        test_data=[[temperature,rh,ws,rain,ffmc,dmc,isi,classes,region]]
        test_data_scaled=scaler_model.transform(test_data)
        result=ridge_model.predict(test_data_scaled)
        return render_template('index.html',result=result[0])
    else:
        return render_template('index.html')



if __name__=='__main__':
    app.run() 