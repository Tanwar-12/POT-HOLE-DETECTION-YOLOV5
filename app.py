from flask import Flask, render_template, request
import cv2
import numpy as np
import torch
import os

app = Flask(__name__)

yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp5/weights/last.pt', force_reload=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['file']
    file_extension = file.filename.rsplit('.', 1)[1].lower()
    
    if file_extension in ['jpg', 'jpeg', 'png']:
        # Image processing
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        results = yolo_model(image)
        cv2.imshow('YOLO', np.squeeze(results.render()))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return "Image detection completed."
    
    elif file_extension in ['mp4', 'avi', 'mov']:
        # Video processing
        video_path = os.path.join(app.root_path, 'uploads', file.filename)
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        file.save(video_path)

        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened(): 
            ret, frame = cap.read()
            
            results = yolo_model(frame)

            cv2.imshow('YOLO', np.squeeze(results.render()))
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
        return "Video detection completed."
    
    else:
        return "Invalid file type. Only images (jpg, jpeg, png) and videos (mp4, avi, mov) are supported."

if __name__ == '__main__':
    app.run()
