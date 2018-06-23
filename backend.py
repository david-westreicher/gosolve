from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import random
import json
from territory import calculate_territory

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

@app.route('/', methods=['GET'])
def index():
    return render_template('choose_file.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    print(request, request.files, request.form)
    if 'file' not in request.files:
        flash('No file part')
        return redirect('/')
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect('/')
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return redirect('/corners/' + filename)

@app.route('/territory/<img>')
def territory(img):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], img)
    corners = request.args['corners'].split(',')
    corners = [float(e) for e in corners]
    corners = list(zip(corners[::2], corners[1::2]))
    gf, area, h, (score_black, score_white) = calculate_territory(img_path, corners)
    json_data = {
        'gf': gf,
        'area': area,
        'h': h,
        'score': {
            'white': score_white,
            'black': score_black,
        },
    }
    for row in gf:
        print(row)
    with open(img_path + '_info.json', 'w') as f:
        json.dump(json_data, f)
    return redirect('/show/' + img)

@app.route('/corners/<filename>')
def corners(filename):
    return render_template('corners.html', img=filename)

@app.route('/show/<img>')
def show(img):
    return render_template('render.html', img=img)

@app.route('/uploads/<filename>')
def send_file(filename):
    print(filename)
    return send_from_directory(UPLOAD_FOLDER, filename)
