import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('/home/bhalewar/brain_tumor_-detection_using_CNN/models/brain tumber model.h5', compile=False)
class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']

UPLOAD_FOLDER = "./uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            img = tf.keras.utils.load_img(file_location, target_size=(256, 256))
            imgarr = tf.keras.utils.img_to_array(img) / 255.0
            imgarr = np.expand_dims(imgarr, axis=0)

            prediction = model.predict(imgarr)
            predctclss = np.argmax(prediction, axis=1)[0]
            confidence_score = np.max(prediction)

            result = class_labels[predctclss]

            return render_template(
                'index.html',
                result=result,
                confidence=f'{confidence_score*100:.2f}%',
                file_path=f'/uploads/{file.filename}',
                file_name = str(file.filename)
            )
    return render_template('index.html', result=None)

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=False)
