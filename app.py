import numpy as np
import requests
import os
import sys
import time
import flask.json
from flask import Flask, send_file, request, jsonify, render_template, send_from_directory, redirect, make_response
from PIL import Image, ImageFilter, ImageDraw
from io import BytesIO
from base64 import b64encode
import uuid
import json
import dlib
import pickle
import codecs


from math import sqrt


def euclidean_dist(vector_x, vector_y):
    if len(vector_x) != len(vector_y):
        raise Exception('Vectors must be same dimensions')
    return sum((vector_x[dim] - vector_y[dim]) ** 2 for dim in range(len(vector_x)))


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(os.path.join(os.path.dirname(
    __file__), 'shape_predictor_68_face_landmarks.dat'))
facerec = dlib.face_recognition_model_v1(os.path.join(os.path.dirname(
    __file__), 'dlib_face_recognition_resnet_model_v1.dat'))


input_height = 128
input_width = 128

app = Flask(__name__)

face_data = {
    'main': []
}
with open(os.path.join(os.path.dirname(
        __file__), 'data.json')) as json_file:
    face_data = json.load(json_file)

print(face_data)


@app.route('/reg', methods=["POST"])
def recognize():
    start_time = time.time()
    upload_time = 0.0
    file = request.files['image']
    if file:
        file.seek(0)
        data = file.read()
        image = Image.open(BytesIO(data))
        width, height = image.size
        size = 1200, height / width * 1200
        image.thumbnail(size, Image.ANTIALIAS)
        img = np.asarray(image)
        dets = detector(img, 1)

        b64_faces = []
        descriptors = []

        draw = ImageDraw.Draw(image)

        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            shape = sp(img, d)
            print(shape)
            face_chip = dlib.get_face_chip(img, shape)
            buffered = BytesIO()
            pil_face = Image.fromarray(face_chip)
            pil_face.save(buffered, format="JPEG")
            img_str = b64encode(buffered.getvalue()).decode("utf-8")
            b64_faces.append(img_str)
            draw.rectangle([d.left(), d.top(), d.right(),
                            d.bottom()], fill=None, width=2)
            face_descriptor = facerec.compute_face_descriptor(face_chip)

            picked = codecs.encode(pickle.dumps(
                face_descriptor), "base64").decode()
            descriptors.append(picked)

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({
            'data': img_str,
            'faces': b64_faces,
            'descriptors': descriptors
        })


user_id = 'main'


@ app.route('/')
def home():
    resp = make_response(render_template(
        'home.html', uuid=user_id, data=face_data[user_id]))
    return resp


@ app.route('/faces')
def faces():
    return jsonify(face_data[user_id])


@ app.route('/save', methods=["POST"])
def save():
    data = request.get_json()
    if (user_id not in face_data):
        face_data[user_id] = []
    face_data[user_id] = data

    with open(os.path.join(os.path.dirname(
            __file__), 'data.json'), 'w') as outfile:
        json.dump(face_data, outfile)

    return jsonify(face_data[user_id])


@app.route('/check', methods=["POST"])
def check():
    file = request.files['image']
    if file:
        file.seek(0)
        data = file.read()
        image = Image.open(BytesIO(data))
        width, height = image.size
        size = 1200, height / width * 1200
        image.thumbnail(size, Image.ANTIALIAS)
        img = np.asarray(image)
        dets = detector(img, 1)

        b64_faces = []
        descriptors = []

        descriptors_by_name = {}
        for data in face_data[user_id]:
            descriptors_by_name[data['name']] = pickle.loads(
                codecs.decode(data['descriptor'].encode(), "base64"))

        draw = ImageDraw.Draw(image)

        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            shape = sp(img, d)
            print(shape)
            face_chip = dlib.get_face_chip(img, shape)
            buffered = BytesIO()
            pil_face = Image.fromarray(face_chip)
            pil_face.save(buffered, format="JPEG")
            img_str = b64encode(buffered.getvalue()).decode("utf-8")
            b64_faces.append(img_str)
            draw.rectangle([d.left(), d.top(), d.right(),
                            d.bottom()], fill=None, width=2)
            face_descriptor = facerec.compute_face_descriptor(face_chip)

            username = "unknown"
            last_match = 1
            for name in descriptors_by_name:
                descriptor = descriptors_by_name[name]
                match = euclidean_dist(descriptor, face_descriptor)
                print("Match %.2f" % match)
                if match < 0.2 and match < last_match:
                    last_match = match
                    username = name

            draw.text((d.left(), d.top() - 10), "%s %.2f" %
                      (username, (1-last_match) * 100.0))

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({
            'data': img_str,
        })


app.run(host='0.0.0.0')
