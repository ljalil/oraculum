import os
import numpy as np
import torch
from flask import Flask, jsonify, request

from wen2018 import CaseWesternClassifier

app = Flask(__name__)

model = CaseWesternClassifier()
model.load_state_dict(torch.load('wen2018-pretrained'))
model.eval()

def transform_signal(infile):
    file = np.loadtxt(infile)
    file = file[:64*64].reshape(1,1,64,64)
    return torch.Tensor(file)

def get_prediction(input_tensor):
    pred = model.forward(input_tensor).argmax(dim=1)
    return pred

@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'Try POSTing to the /predict endpoint with a plain text file attachment containing the vibration signal'})

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file', None)
        if file is not None:
            input_tensor = transform_signal(file.filename)
            prediction_idx = get_prediction(input_tensor)
            print(prediction_idx)
            return jsonify({'class': prediction_idx.item()})
        else:
            return "No data provided"


if __name__ == '__main__':
    app.run()


