import pickle
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas

app = Flask(__name__)
model = pickle.load(open('flight_price_prediction.pkl', 'rb'))
#scaler = pickle.load(open('extra_trees_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        new_data = np.array(list(data.values())).reshape(1, -1)
        output = model.predict(new_data)
        return jsonify(output[0])
    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(debug=True)
    