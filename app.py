import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fallback_secret')
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = r"C:\Users\Lenovo\Desktop\Actionfind\final_action_recognition\convlstm_model.h5"
try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Model Loaded Successfully")
    else:
        print("❌ Model file not found")
        model = None
except Exception as e:
    print("❌ Model Load Error:", e)
    model = None

CLASSES_LIST = ['BaseballPitch', 'Basketball', 'BenchPress', 'Biking', 'Billiards']

SEQUENCE_LENGTH = 20
IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(video_path):
    if model is None:
        return "Model Not Loaded", 0.0
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(total_frames // SEQUENCE_LENGTH, 1)

    for i in range(SEQUENCE_LENGTH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip)
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH)) / 255.0
        frames.append(frame)
    cap.release()

    while len(frames) < SEQUENCE_LENGTH:
        frames.append(frames[-1] if frames else np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3)))

    input_data = np.expand_dims(frames, axis=0)

    try:
        pred = model.predict(input_data)[0]
        label = np.argmax(pred)
        confidence = float(pred[label]) * 100
        return CLASSES_LIST[label], confidence
    except:
        return "Prediction Error", 0.0

@app.route('/', methods=['GET', 'POST'])
def home():
    action = None
    confidence = None
    filename = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            action, confidence = process_video(file_path)
    return render_template('index.html', action=action, confidence=confidence, filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
