from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
with open('modelnew.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
app = Flask(__name__,static_url_path='/static')
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/index')
def index():
    return render_template('index.html')
@app.route('/index', methods=['POST'])
def index_predict():
    features = [request.form.get('feature1'), request.form.get('feature2'), request.form.get('feature3'), request.form.get('feature4'), request.form.get('feature5'), request.form.get('feature6'), request.form.get('feature7'), request.form.get('feature8')]
    if '' in features:
        return render_template('index.html', error='Please fill in all fields.')
    try:
        features = [float(x) for x in features]
    except ValueError:
        return render_template('index.html', error='Invalid input. Please enter numeric values.')
    input_data = np.array([features])
    prediction = model.predict(scaler.transform(input_data))
    return render_template('index.html', prediction=prediction[0])

@app.route('/about', methods=['GET','POST'])
def about():
    return render_template('about.html')
if __name__ == '__main__':
    app.run(debug=True)

