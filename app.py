import os
import cv2
import numpy as np
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import mediapipe as mp
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# =====================================================
# چہرے کے 3D landmarks نکالیں
# =====================================================
def extract_face_data(image_bytes):
    # Image decode
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]

        # Landmarks کو 3D coordinates میں convert
        points_3d = []
        for lm in landmarks.landmark:
            points_3d.append({
                'x': lm.x,  # 0-1 normalized
                'y': lm.y,  # 0-1 normalized
                'z': lm.z   # depth
            })

        # تصویر کو base64 میں convert
        img_pil = Image.fromarray(img_rgb)
        
        # Resize for web
        max_size = 512
        if max(img_pil.width, img_pil.height) > max_size:
            img_pil.thumbnail((max_size, max_size), Image.LANCZOS)
        
        buffered = BytesIO()
        img_pil.save(buffered, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return {
            'landmarks': points_3d,
            'imageBase64': f'data:image/jpeg;base64,{img_b64}',
            'width': img_pil.width,
            'height': img_pil.height,
            'originalWidth': w,
            'originalHeight': h
        }

# =====================================================
# Routes
# =====================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_face():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    image_bytes = file.read()

    result = extract_face_data(image_bytes)

    if result is None:
        return jsonify({'error': 'No face detected. Please use a clear front-facing photo.'}), 400

    return jsonify(result)

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
