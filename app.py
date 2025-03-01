import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load ML Model
model_path = os.path.abspath("newmodel.h5")
try:
    model = tf.keras.models.load_model(model_path, compile=False)
    print("✅ Model Loaded Successfully")
except Exception as model_error:
    print(f"❌ Model Load Error: {model_error}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ✅ Process Video Function
def process_video(video_path):
    if model is None:
        return "Model not loaded", 0

    cap = cv2.VideoCapture(video_path)
    frames = []
    max_frames = 20

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames)
    frame_count = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % step == 0 and len(frames) < max_frames:
            frame = cv2.resize(frame, (128, 128)) / 255.0
            frames.append(frame)
        frame_count += 1
    cap.release()

    while len(frames) < max_frames:
        frames.append(frames[-1])

    frames = np.expand_dims(np.array(frames), axis=0)

    try:
        prediction = model.predict(frames)
        return map_prediction_to_label(np.argmax(prediction)), float(np.max(prediction)) * 100
    except Exception as e:
        return f'Prediction Error: {str(e)}', 0

# ✅ Map Predictions to Labels
def map_prediction_to_label(predicted_class):
    action_labels = {0: "TaiChi", 1: "HorseRace", 2: "WalkingWithDog", 3: "BaseballPitch"}
    return action_labels.get(predicted_class, "Unknown Action")

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_location)

            # Process the video
            action, confidence = process_video(file_location)

            # Render the template with the video file
            return render_template('upload_video.html', action=action, confidence=f"{confidence:.2f}%", filename=filename)

    return render_template('upload_video.html', action=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
