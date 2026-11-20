<img width="1366" height="636" alt="Screenshot (41)" src="https://github.com/user-attachments/assets/e0c357a5-2876-49f0-9aca-459ca83775d9" />


🎥 Action Recognition using Deep Learning

This project demonstrates Human Action Recognition using deep learning models. It combines Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM/ConvLSTM) to process spatio-temporal features from video sequences and classify different human actions.

📌 Features

✅ Train & test deep learning models (CNN + LSTM/ConvLSTM).

✅ Flask-based web app to upload and predict actions on videos.

✅ Pre-trained model (convlstm_model.h5) included.

✅ Custom dataset with multiple action classes.

✅ Supports .mp4 and .avi video formats.

🧠 Tech Stack

Language: Python

Frameworks & Libraries:

TensorFlow / Keras

OpenCV

NumPy, Pandas

Flask (for web interface)

Frontend: HTML, CSS (Flask templates)


🚀 Getting Started
1. Clone the repository
git clone https://github.com/Anush0104/Action-Recognition.git
cd Action-Recognition

2. Create & activate virtual environment (recommended)
python -m venv venv
venv\Scripts\activate     # Window

source venv/bin/activate  # Mac/Linux

4. Install dependencies
pip install -r requirements.txt



4. Run the Flask app
python app.py


Go to 👉 http://127.0.0.1:5000 in your browser.

🎯 Usage

Open the web app.

Upload a video file (.mp4 or .avi).

The model predicts the action category.

📊 Dataset

Custom dataset created with at least 15+ videos per action.

Contains multiple human action classes such as:

Typically trimmed UCF 50 dataset 

Achieved Accuracy: ~95 %
📌 Future Improvements

Integrate Real-time webcam action recognition.

Add more action classes for better generalization.

