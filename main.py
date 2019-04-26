from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

loaded_model = pickle.load(open('income_tree_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,11)   
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/result', methods = ['POST'])

def result():
    prediction = ''
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        print(to_predict_list.values())
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        print('Before sending to model', to_predict_list)
        result = ValuePredictor(to_predict_list)
        print('result from model', result)
        if int(result) == 0:
            prediction = 'You are predicted to earn "Less than USD50,000"'
        elif int(result) == 1:
            prediction = 'You are predicted to earn "More than USD50,000"'
        else:
            prediction = 'Unknown Class'
        print(prediction)

        return render_template('result.html', prediction = prediction)


if __name__ == "__main__":
    app.run(debug=True)
