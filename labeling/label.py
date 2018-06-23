from flask import Flask, render_template, request
import time
import os
import random

app = Flask(__name__)

@app.route('/')
def index():
    img = random.choice(os.listdir('static/unknown/'))
    img = f'static/unknown/{img}'
    print(img)
    return render_template('label.html', image_path=img)

@app.route('/label', methods = ['POST'])
def label():
    img = request.args['img']
    l = request.args['label']
    os.rename(img, img.replace('unknown', l))
    print('label', img, l)
    return ('', 204)
