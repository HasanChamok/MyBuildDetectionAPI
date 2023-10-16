import os
import cv2 as cv
import torch
import numpy as np
import tensorflow as tf
from tkinter import image_names
from unittest import result
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = Flask(__name__)

UPLOAD_FOLDER = 'saved'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = torch.hub.load('ultralytics/yolov5','custom', path='best.pt')
classes = []
with open('classes.txt','r') as f:
    for line in f:
        classes.append(line.strip())
        
def detect(img_path):
    label , confidence = [] , []
    
    img = cv.imread(img_path)
    img = cv.resize(img,(416,416))
    
    results = model(img)
    
    for res in results.pandas().xyxy:
        # print(len(res))
        for obj in range(len(res)):
            className = classes[res['class'][obj]]
            label.append(className)
            confidence.append(res['confidence'][obj])
    
    return label , confidence 



@app.route('/upload', methods=['GET','POST'])

def upload_file():
    label , confidence = [] , []
    if request.method == 'POST':
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        # img = request.files['image']
        # vid = request.files['video']
        
        # if img.filename == '' and vid.filename == '':
        #     return 'No selected files'
        
        #Handle image files
        # if img.filename != '':
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return jsonify({'message': 'File uploaded successfully', 'filename': filename})
        else:
            return jsonify({'error': 'Invalid file format'})
        
        # img_name = secure_filename(file.filename)
        # img_path = 'saved/' + img_name
        # img.save(img_path)
        
        # img = cv.imread(img_path)
        # img = cv.resize(img,(416,416))
        
        # label, confidence = detect(img_path)
            # label , confidence = ['BALL'] , ["0.56"]
            
    return jsonify({"Class": label},{"Confidence Score":confidence})
        
        


if __name__ == '__main__':
    app.run(debug=True)
