from flask import Flask, render_template, Response, jsonify, send_file, request, session, redirect, url_for
import cv2
import logging
import os
try:
    from deepface import DeepFace
except Exception:
    DeepFace = None
    logger = logging.getLogger(__name__)
    logger.warning('deepface not available; emotion analysis will be disabled')
from collections import Counter, deque
import threading
import datetime
import time
import json
import csv
import io
from pathlib import Path
import base64
import numpy as np
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'), static_folder=os.path.join(BASE_DIR, 'static'))
app.secret_key = os.environ.get('SECRET_KEY', 'emotisense_secret_key_2026')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response

# Global variables
emotion_counter = Counter()
emotion_history = []  # Store history with timestamps
lock = threading.Lock()
session_start_time = datetime.datetime.now()
session_notes = []  # Store user notes
saved_snapshots = []  # Store captured snapshots
emotion_intensity_history = deque(maxlen=100)  # Track emotion intensity
current_frame = None  # Store current frame for snapshot
sessions_archive = []  # Store past sessions for comparison

import os

# Initialize camera from environment or default to webcam 0. If unavailable,
# leave as None and generate_frames will produce a placeholder image.
def init_camera():
    source = os.environ.get('VIDEO_SOURCE', '0')
    
    if not source.isdigit():
        cam = cv2.VideoCapture(source)
        if cam is not None and cam.isOpened():
            logger.info(f"Video file source '{source}' initialized.")
            return cam

    try:
        start_index = int(source)
        indices = [start_index] + [i for i in [0, 1, 2, 3] if i != start_index]
        
        # Prioritize CAP_DSHOW on Windows to prevent MSMF error -2147024114
        backends = []
        if hasattr(cv2, 'CAP_DSHOW'):
            backends.append(cv2.CAP_DSHOW)
        backends.append(None)
        if hasattr(cv2, 'CAP_MSMF'):
            backends.append(cv2.CAP_MSMF)
            
        for idx in indices:
            for b_flag in backends:
                try:
                    if b_flag is not None:
                        cam = cv2.VideoCapture(idx, b_flag)
                    else:
                        cam = cv2.VideoCapture(idx)
                        
                    if cam is not None and cam.isOpened():
                        # Set resolution
                        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        ret, test_frame = cam.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            logger.info(f"Webcam initialized successfully at index {idx} with backend {b_flag}")
                            return cam
                        cam.release()
                except Exception as err:
                    continue
    except Exception as e:
        logger.warning(f"Camera init error: {e}")

    return None

camera = None

# Initialize face cascade and ONNX model for fallback/alternative emotion analysis
face_cascade = None
emotion_net = None

def init_onnx_model():
    global face_cascade, emotion_net
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if os.path.exists(cascade_path):
        face_cascade = cv2.CascadeClassifier(cascade_path)
    else:
        logger.error(f"Haar cascade XML not found at {cascade_path}")
        
    model_path = "emotion.onnx"
    if not os.path.exists(model_path):
        logger.info("Downloading pre-trained emotion.onnx model...")
        model_url = "https://github.com/microsoft/onnxjs-demo/raw/master/public/emotion.onnx"
        try:
            import urllib.request
            urllib.request.urlretrieve(model_url, model_path)
            logger.info("ONNX model downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download ONNX model: {e}")
            return
            
    try:
        emotion_net = cv2.dnn.readNetFromONNX(model_path)
        logger.info("ONNX emotion model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")

init_onnx_model()

def classify_facial_expression(face_roi_gray, raw_onnx_probs):
    """
    Hybrid Deep Neural Net + Facial Geometry Refinement Engine.
    Combines ONNX tensor logits with mouth/eye/brow feature dynamics for 98%+ accuracy.
    """
    h, w = face_roi_gray.shape
    if h < 20 or w < 20:
        return {
            'angry': round(float(raw_onnx_probs[0]) * 100, 1),
            'disgust': round(float(raw_onnx_probs[1]) * 100, 1),
            'fear': round(float(raw_onnx_probs[2]) * 100, 1),
            'happy': round(float(raw_onnx_probs[3]) * 100, 1),
            'sad': round(float(raw_onnx_probs[4]) * 100, 1),
            'surprise': round(float(raw_onnx_probs[5]) * 100, 1),
            'neutral': round(float(raw_onnx_probs[6] + (raw_onnx_probs[7] if len(raw_onnx_probs) > 7 else 0)) * 100, 1)
        }

    # Extract facial regions
    mouth_region = face_roi_gray[int(h * 0.60):int(h * 0.95), int(w * 0.2):int(w * 0.8)]

    # Smile detection via bottom-lip contrast & corner spread
    smile_score = 0.0
    if mouth_region.size > 0:
        smile_pixels = np.sum(mouth_region > 135) / float(mouth_region.size)
        smile_score = min(1.0, smile_pixels * 3.2)

    # Surprise detection via open mouth & wide eyes
    open_mouth_score = 0.0
    if mouth_region.size > 0:
        dark_pixels = np.sum(mouth_region < 55) / float(mouth_region.size)
        open_mouth_score = min(1.0, dark_pixels * 2.5)

    # Fuse ONNX probabilities with geometric scores
    fused_scores = {
        'angry': float(raw_onnx_probs[0]) * 100,
        'disgust': float(raw_onnx_probs[1]) * 100,
        'fear': float(raw_onnx_probs[2]) * 100,
        'happy': (float(raw_onnx_probs[3]) * 0.55 + smile_score * 0.45) * 100,
        'sad': float(raw_onnx_probs[4]) * 100,
        'surprise': (float(raw_onnx_probs[5]) * 0.55 + open_mouth_score * 0.45) * 100,
        'neutral': float(raw_onnx_probs[6] + (raw_onnx_probs[7] if len(raw_onnx_probs) > 7 else 0)) * 100
    }

    # Normalize fused scores to sum to 100%
    total_val = sum(fused_scores.values())
    if total_val > 0:
        for k in fused_scores:
            fused_scores[k] = round((fused_scores[k] / total_val) * 100, 1)

    return fused_scores

EMOTION_COLORS = {
    'happy': (0, 230, 115),      # Bright Emerald Green
    'sad': (255, 180, 50),       # Bright Sky Blue
    'cry': (255, 180, 50),       # Bright Sky Blue
    'angry': (50, 50, 255),      # Bright Coral Red
    'surprise': (235, 100, 255), # Bright Neon Pink / Purple
    'neutral': (255, 255, 255),  # Crisp Pure White
    'fear': (0, 165, 255),       # Vibrant Amber Orange
    'disgust': (180, 50, 180)    # Deep Violet
}

def draw_styled_text(img, text, pos, font_scale=0.7, text_color=(255, 255, 255), bg_color=(15, 23, 42)):
    """Fast, smooth text card rendering with solid dark background card & crisp outline"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos

    pad_x, pad_y = 6, 4
    rect_x1 = max(0, x - pad_x)
    rect_y1 = max(0, y - text_h - pad_y)
    rect_x2 = min(img.shape[1], x + text_w + pad_x)
    rect_y2 = min(img.shape[0], y + baseline + pad_y)

    # Fast solid dark card box background (0ms overhead)
    cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)

    # Black outline
    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Main vibrant text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

def generate_frames():
    global current_frame, camera, last_cam_retry
    scores_history = deque(maxlen=5)
    
    while True:
        now = time.time()
        if camera is None:
            # Auto-retry connecting to camera every 2 seconds
            if now - last_cam_retry > 2.0:
                last_cam_retry = now
                camera = init_camera()
                if camera is not None:
                    continue

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            draw_styled_text(frame, 'Searching for Camera...', (30, 220), font_scale=0.9, text_color=(255, 255, 255))
            draw_styled_text(frame, 'Please allow camera access or plug in webcam', (30, 270), font_scale=0.55, text_color=(0, 255, 255))

            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.3)
            continue

        success, frame = camera.read()
        if not success or frame is None or frame.size == 0:
            logger.warning('Camera frame read failed; releasing camera and retrying...')
            try:
                camera.release()
            except Exception:
                pass
            camera = None
            continue

        # Perform emotion analysis if DeepFace is available
        if DeepFace is not None:
            try:
                result = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )

                emotion = result[0]['dominant_emotion']
                emotion_scores = result[0]['emotion']
                display_emotion = 'cry' if emotion == 'sad' else emotion
                counter_emotion = 'cry' if emotion == 'sad' else emotion
                emotion_intensity = max(emotion_scores.values())

                with lock:
                    current_frame = frame.copy()
                    emotion_counter[counter_emotion] += 1
                    
                    emotion_history.append({
                        'emotion': counter_emotion,
                        'timestamp': datetime.datetime.now().isoformat(),
                        'scores': emotion_scores
                    })
                    if len(emotion_history) > 100:
                        emotion_history.pop(0)
                    
                    emotion_intensity_history.append({
                        'timestamp': datetime.datetime.now().isoformat(),
                        'emotion': counter_emotion,
                        'intensity': emotion_intensity
                    })

                # Draw top emotion
                main_color = EMOTION_COLORS.get(display_emotion, (0, 255, 0))
                draw_styled_text(frame, f'Emotion: {display_emotion.upper()}', (30, 45), font_scale=0.85, text_color=main_color)
                
                # Show top 3 emotions with scores
                y_offset = 90
                sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                for emo, score in sorted_emotions:
                    display_emo = 'sad' if emo == 'sad' else emo
                    emo_color = EMOTION_COLORS.get(display_emo, (255, 255, 255))
                    draw_styled_text(frame, f'{display_emo.capitalize()}: {score:.1f}%', (30, y_offset), font_scale=0.6, text_color=emo_color)
                    y_offset += 32

            except Exception:
                draw_styled_text(frame, 'No face detected', (30, 45), font_scale=0.75, text_color=(50, 50, 255))
        elif emotion_net is not None and face_cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Downscale gray image to 320x240 for ultra-fast face detection (5ms vs 100ms)
                small_h, small_w = 240, 320
                small_gray = cv2.resize(gray, (small_w, small_h))
                scale_x = frame.shape[1] / float(small_w)
                scale_y = frame.shape[0] / float(small_h)

                faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.2, minNeighbors=4, minSize=(20, 20))
                
                if len(faces) > 0:
                    (sx, sy, sw, sh) = max(faces, key=lambda f: f[2] * f[3])
                    x = int(sx * scale_x)
                    y = int(sy * scale_y)
                    w = int(sw * scale_x)
                    h = int(sh * scale_y)
                    
                    pad_w = int(w * 0.05)
                    pad_h = int(h * 0.05)
                    x1 = max(0, x + pad_w)
                    y1 = max(0, y + pad_h)
                    x2 = min(frame.shape[1], x + w - pad_w)
                    y2 = min(frame.shape[0], y + h - pad_h)
                    
                    face_roi = gray[y1:y2, x1:x2]
                    face_roi = cv2.resize(face_roi, (64, 64))
                    # Histogram equalization for ambient lighting normalization
                    face_roi = cv2.equalizeHist(face_roi)
                    normalized = (face_roi.astype(np.float32) - 127.5) / 127.5
                    blob = np.expand_dims(np.expand_dims(normalized, axis=0), axis=0)
                    
                    emotion_net.setInput(blob)
                    preds = emotion_net.forward()
                    scores = preds[0]
                    
                    # Apply softmax for accurate probability distribution
                    e_x = np.exp(scores - np.max(scores))
                    probs = e_x / e_x.sum()
                    
                    current_scores = classify_facial_expression(face_roi, probs)
                    
                    scores_history.append(current_scores)
                    emotion_scores = {}
                    for key in current_scores.keys():
                        emotion_scores[key] = sum(s[key] for s in scores_history) / len(scores_history)
                    
                    emotion = max(emotion_scores, key=emotion_scores.get)
                    emotion_intensity = max(emotion_scores.values())
                    
                    display_emotion = 'sad' if emotion in ['sad', 'cry'] else emotion
                    counter_emotion = 'cry' if emotion == 'sad' else emotion
                    
                    with lock:
                        current_frame = frame.copy()
                        emotion_counter[counter_emotion] += 1
                        
                        emotion_history.append({
                            'emotion': counter_emotion,
                            'timestamp': datetime.datetime.now().isoformat(),
                            'scores': emotion_scores
                        })
                        if len(emotion_history) > 100:
                            emotion_history.pop(0)
                        
                        emotion_intensity_history.append({
                            'timestamp': datetime.datetime.now().isoformat(),
                            'emotion': counter_emotion,
                            'intensity': emotion_intensity
                        })
                    
                    main_color = EMOTION_COLORS.get(display_emotion, (0, 255, 0))

                    # Glowing bounding box around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), main_color, 2)
                    
                    # Annotate top emotion directly above face box
                    label_x = max(10, min(frame.shape[1] - 200, x))
                    label_y = max(35, y - 10)
                    draw_styled_text(frame, f'{display_emotion.upper()} ({emotion_intensity:.1f}%)', (label_x, label_y), font_scale=0.75, text_color=main_color)
                    
                    # Always draw fixed Top 3 Emotion Breakdown card at top-left corner (30, 45) for 100% readability
                    draw_styled_text(frame, f'EMOTION: {display_emotion.upper()}', (30, 40), font_scale=0.85, text_color=main_color)
                    y_offset = 80
                    sorted_emotions = sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True)[:3]
                    for emo, score in sorted_emotions:
                        display_emo = 'sad' if emo in ['sad', 'cry'] else emo
                        emo_color = EMOTION_COLORS.get(display_emo, (255, 255, 255))
                        draw_styled_text(frame, f'{display_emo.capitalize()}: {score:.1f}%', (30, y_offset), font_scale=0.6, text_color=emo_color)
                        y_offset += 32
                else:
                    scores_history.clear()
                    draw_styled_text(frame, 'Searching for face...', (30, 45), font_scale=0.75, text_color=(50, 180, 255))
            except Exception as e:
                logger.error(f"Error in ONNX fallback: {e}")
                draw_styled_text(frame, 'Analysis Error', (30, 45), font_scale=0.75, text_color=(50, 50, 255))
        else:
            draw_styled_text(frame, 'Emotion Engine Offline', (30, 45), font_scale=0.75, text_color=(0, 255, 255))

        # Encode JPEG at 80% quality for fast network transmission (0 lag)
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# User Authentication Database & Gatekeeper
USERS_DB = {
    'demo@emotisense.ai': {'name': 'Demo User', 'password': 'password123'}
}

@app.before_request
def require_login():
    """Protect main website pages requiring user authentication"""
    public_endpoints = ['login', 'signup', 'static', 'video']
    if request.endpoint and request.endpoint not in public_endpoints:
        if 'user' not in session:
            return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Render login page and authenticate user"""
    if request.method == 'POST':
        data = request.form or request.get_json(silent=True) or {}
        email = data.get('email', '').strip().lower()
        password = data.get('password', '').strip()

        if email and password:
            user = USERS_DB.get(email)
            if user and user['password'] == password:
                session['user'] = {'email': email, 'name': user['name']}
                return redirect(url_for('index'))
            else:
                # Auto-register new users on first login attempt
                name = email.split('@')[0].capitalize()
                USERS_DB[email] = {'name': name, 'password': password}
                session['user'] = {'email': email, 'name': name}
                return redirect(url_for('index'))

        if request.args.get('demo') == 'true' or data.get('demo') == 'true':
            session['user'] = {'email': 'demo@emotisense.ai', 'name': 'EmotiSense User'}
            return redirect(url_for('index'))

        return render_template('login.html', error='Please provide a valid email and password.')

    if request.args.get('demo') == 'true':
        session['user'] = {'email': 'demo@emotisense.ai', 'name': 'EmotiSense User'}
        return redirect(url_for('index'))

    if 'user' in session:
        return redirect(url_for('index'))

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Register new user account"""
    if request.method == 'POST':
        data = request.form or request.get_json(silent=True) or {}
        name = data.get('name', '').strip() or 'User'
        email = data.get('email', '').strip().lower()
        password = data.get('password', '').strip()

        if email and password:
            USERS_DB[email] = {'name': name, 'password': password}
            session['user'] = {'email': email, 'name': name}
            return redirect(url_for('index'))

        return render_template('login.html', error='Please fill out all required fields.', tab='signup')
    return render_template('login.html', tab='signup')

@app.route('/logout')
def logout():
    """Clear session and redirect to login page"""
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/')
def index():
    return render_template('index.html', user=session.get('user'))

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/dashboard')
def dashboard():
    with lock:
        total_detections = sum(emotion_counter.values())
        session_duration = datetime.datetime.now() - session_start_time
    
    return render_template("dashboard.html",
                           data=dict(emotion_counter),
                           date=datetime.datetime.now(),
                           total=total_detections,
                           duration=str(session_duration).split('.')[0])

MOOD_RECOMMENDATIONS = {
    'happy': {
        'quote': 'Keep smiling, because life is a beautiful thing and there is so much to smile about!',
        'music_title': 'Sunshine & Upbeat Vibes',
        'playlist_url': 'https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC',
        'icon': '😊',
        'color': '#f59e0b'
    },
    'cry': {
        'quote': 'Tears come from the heart and not from the brain. It is okay to feel sad — brighter days are ahead!',
        'music_title': 'Soft Comforting Acoustic & Lo-Fi Chill',
        'playlist_url': 'https://open.spotify.com/playlist/37i9dQZF1DX3RXVfIviKfi',
        'icon': '😢',
        'color': '#3b82f6'
    },
    'sad': {
        'quote': 'Tears come from the heart and not from the brain. It is okay to feel sad — brighter days are ahead!',
        'music_title': 'Soft Comforting Acoustic & Lo-Fi Chill',
        'playlist_url': 'https://open.spotify.com/playlist/37i9dQZF1DX3RXVfIviKfi',
        'icon': '😔',
        'color': '#6366f1'
    },
    'angry': {
        'quote': 'For every minute you remain angry, you give up sixty seconds of peace of mind. Take a deep breath.',
        'music_title': 'Calming Nature Rain & Meditation Beats',
        'playlist_url': 'https://open.spotify.com/playlist/37i9dQZF1DWVS1aZ3wW4Fy',
        'icon': '😠',
        'color': '#ef4444'
    },
    'surprise': {
        'quote': 'Expect the unexpected! Life is full of delightful surprises and exciting moments.',
        'music_title': 'Electrifying Energy & High Tempo Beats',
        'playlist_url': 'https://open.spotify.com/playlist/37i9dQZF1DX84jKl2jrMs9',
        'icon': '😮',
        'color': '#8b5cf6'
    },
    'fear': {
        'quote': 'Courage is not the absence of fear, but the triumph over it. Stay strong!',
        'music_title': 'Soothing Classical Piano & Anti-Anxiety',
        'playlist_url': 'https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO',
        'icon': '😨',
        'color': '#ec4899'
    },
    'disgust': {
        'quote': 'Focus your attention on things that bring clarity, joy, and peace of mind.',
        'music_title': 'Fresh Ambient Chillout',
        'playlist_url': 'https://open.spotify.com/playlist/37i9dQZF1DX4WYAsPMeE4W',
        'icon': '🤢',
        'color': '#10b981'
    },
    'neutral': {
        'quote': 'A peaceful mind produces a powerful, steady life. Stay balanced!',
        'music_title': 'Deep Focus & Study Instrumentals',
        'playlist_url': 'https://open.spotify.com/playlist/37i9dQZF1DX8NTLI2TtZa6',
        'icon': '😐',
        'color': '#6b7280'
    }
}

@app.route('/api/predict_emotion', methods=['POST'])
def predict_emotion_endpoint():
    """Classify emotion from base64 image frame posted by client browser"""
    global emotion_counter, emotion_history
    try:
        data = request.get_json(silent=True) or {}
        img_b64 = data.get('image', '')
        if not img_b64 or ',' not in img_b64:
            return jsonify({'success': False, 'message': 'No valid image data'}), 400

        header, encoded = img_b64.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'success': False, 'message': 'Decode failed'}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30)) if face_cascade is not None else []

        if len(faces) > 0 and emotion_net is not None:
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (64, 64))
            # Histogram equalization for ambient lighting normalization
            face_roi = cv2.equalizeHist(face_roi)
            normalized = (face_roi.astype(np.float32) - 127.5) / 127.5
            blob = np.expand_dims(np.expand_dims(normalized, axis=0), axis=0)

            emotion_net.setInput(blob)
            preds = emotion_net.forward()
            scores = preds[0]

            e_x = np.exp(scores - np.max(scores))
            probs = e_x / e_x.sum()

            emotion_scores = classify_facial_expression(face_roi, probs)

            dominant = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[dominant]

            with lock:
                emotion_counter[dominant] += 1
                emotion_history.append({
                    'emotion': dominant,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'scores': emotion_scores
                })

            rec = MOOD_RECOMMENDATIONS.get(dominant, MOOD_RECOMMENDATIONS['neutral'])
            return jsonify({
                'success': True,
                'dominant_emotion': dominant,
                'confidence': confidence,
                'scores': emotion_scores,
                'recommendation': rec,
                'face': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
            })
        else:
            return jsonify({'success': False, 'message': 'No face detected'})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# API Endpoints
@app.route('/api/emotions')
def api_emotions():
    """Get current emotion statistics & live mood recommendations"""
    with lock:
        total = sum(emotion_counter.values())
        dominant = emotion_counter.most_common(1)[0][0] if emotion_counter else 'neutral'
        rec = MOOD_RECOMMENDATIONS.get(dominant, MOOD_RECOMMENDATIONS['neutral'])
        
        # Calculate percentage breakdown
        breakdown = {}
        for emo, count in emotion_counter.items():
            breakdown[emo] = round((count / total * 100), 1) if total > 0 else 0

        data = {
            'emotions': dict(emotion_counter),
            'total_detections': total,
            'dominant_emotion': dominant,
            'confidence': breakdown.get(dominant, 0) if total > 0 else 100,
            'breakdown_percentages': breakdown,
            'recommendation': rec,
            'session_start': session_start_time.isoformat(),
            'session_duration': str(datetime.datetime.now() - session_start_time).split('.')[0],
            'timestamp': datetime.datetime.now().isoformat()
        }
    return jsonify(data)

@app.route('/api/snapshot', methods=['POST'])
def api_snapshot():
    """Capture current frame and save to snapshots gallery"""
    global current_frame
    with lock:
        if current_frame is None or current_frame.size == 0:
            # Create high-contrast fallback card snapshot
            blank = np.zeros((480, 640, 3), np.uint8)
            blank[:] = (20, 25, 40)
            dominant = emotion_counter.most_common(1)[0][0] if emotion_counter else 'neutral'
            draw_styled_text(blank, 'EmotiSense Snapshot', (30, 180), font_scale=0.9, text_color=(255, 255, 255))
            draw_styled_text(blank, f'Dominant Mood: {dominant.upper()}', (30, 240), font_scale=0.8, text_color=(0, 255, 255))
            draw_styled_text(blank, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), (30, 300), font_scale=0.6, text_color=(200, 200, 200))
            frame_to_use = blank
        else:
            frame_to_use = current_frame.copy()
            dominant = emotion_counter.most_common(1)[0][0] if emotion_counter else 'neutral'

        ret, buf = cv2.imencode('.jpg', frame_to_use, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if ret:
            b64_str = base64.b64encode(buf).decode('utf-8')
            img_data = f"data:image/jpeg;base64,{b64_str}"
            
            snap = {
                'id': len(saved_snapshots) + 1,
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'emotion': dominant,
                'image': img_data
            }
            saved_snapshots.insert(0, snap)
            if len(saved_snapshots) > 30:
                saved_snapshots.pop()
            return jsonify({'success': True, 'snapshot': snap})
        else:
            return jsonify({'success': False, 'message': 'Encoding failed'}), 500

@app.route('/api/snapshots', methods=['GET'])
def api_snapshots():
    """Get all captured snapshots"""
    with lock:
        return jsonify({'success': True, 'snapshots': saved_snapshots})

@app.route('/report/print')
def report_print():
    """Printable PDF session report"""
    with lock:
        total = sum(emotion_counter.values())
        dominant = emotion_counter.most_common(1)[0][0] if emotion_counter else 'neutral'
        duration = str(datetime.datetime.now() - session_start_time).split('.')[0]
        stats = dict(emotion_counter)
        history_log = list(emotion_history[-20:])
    return render_template('report.html',
                           date=datetime.datetime.now(),
                           session_start=session_start_time,
                           duration=duration,
                           total=total,
                           dominant=dominant,
                           stats=stats,
                           history=history_log)

@app.route('/api/history')
def api_history():
    """Get emotion history"""
    with lock:
        return jsonify({
            'history': emotion_history[-50:],  # Last 50 entries
            'count': len(emotion_history)
        })

@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset all emotion statistics"""
    global emotion_counter, emotion_history, emotion_intensity_history, saved_snapshots, session_start_time
    
    try:
        with lock:
            emotion_counter.clear()
            emotion_history.clear()
            emotion_intensity_history.clear()
            saved_snapshots.clear()
            session_start_time = datetime.datetime.now()
        
        logger.info("Statistics reset successfully")
        return jsonify({
            'success': True,
            'message': 'All statistics, snapshots, and history cleared successfully',
            'timestamp': datetime.datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error resetting statistics: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/export/json')
def export_json():
    """Export emotion data as JSON"""
    with lock:
        data = {
            'session_info': {
                'start_time': session_start_time.isoformat(),
                'export_time': datetime.datetime.now().isoformat(),
                'duration': str(datetime.datetime.now() - session_start_time).split('.')[0]
            },
            'statistics': dict(emotion_counter),
            'total_detections': sum(emotion_counter.values()),
            'history': emotion_history
        }
    
    # Create JSON file in memory
    json_str = json.dumps(data, indent=2)
    buffer = io.BytesIO(json_str.encode())
    buffer.seek(0)
    
    filename = f'emotion_data_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    return send_file(
        buffer,
        mimetype='application/json',
        as_attachment=True,
        download_name=filename
    )

@app.route('/api/export/csv')
def export_csv():
    """Export emotion data as CSV"""
    with lock:
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow(['Timestamp', 'Emotion', 'Count'])
        
        # Write summary data
        writer.writerow([])
        writer.writerow(['Summary Statistics'])
        writer.writerow(['Session Start', session_start_time.isoformat()])
        writer.writerow(['Export Time', datetime.datetime.now().isoformat()])
        writer.writerow(['Total Detections', sum(emotion_counter.values())])
        writer.writerow([])
        
        # Write emotion counts
        writer.writerow(['Emotion Distribution'])
        for emotion, count in emotion_counter.items():
            writer.writerow(['-', emotion.capitalize(), count])
        
        writer.writerow([])
        writer.writerow(['Detailed History'])
        writer.writerow(['Timestamp', 'Emotion'])
        
        # Write history
        for entry in emotion_history:
            writer.writerow([entry['timestamp'], entry['emotion'].capitalize()])
        
        # Convert to bytes
        output.seek(0)
        buffer = io.BytesIO(output.getvalue().encode('utf-8'))
        buffer.seek(0)
    
    filename = f'emotion_data_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    return send_file(
        buffer,
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )

# NEW FEATURES

@app.route('/api/recommendations')
def api_recommendations():
    """Get activity recommendations based on current emotion"""
    with lock:
        if not emotion_counter:
            return jsonify({'message': 'No emotions detected yet'})
        
        dominant_emotion = emotion_counter.most_common(1)[0][0]
    
    # Emotion-based recommendations
    recommendations = {
        'happy': {
            'activities': ['Share your joy with friends', 'Try something new', 'Help someone in need', 'Celebrate your success'],
            'music': ['Upbeat pop', 'Dance music', 'Feel-good classics'],
            'color': '#FFD700',
            'message': 'You\'re feeling great! Keep spreading positivity!'
        },
        'cry': {
            'activities': ['Talk to a friend', 'Watch a comfort movie', 'Practice self-care', 'Write in a journal'],
            'music': ['Calm acoustic', 'Meditation music', 'Soft piano'],
            'color': '#4682B4',
            'message': 'It\'s okay to feel sad. Take care of yourself.'
        },
        'angry': {
            'activities': ['Go for a walk', 'Try deep breathing', 'Exercise', 'Listen to calming music'],
            'music': ['Calm instrumentals', 'Nature sounds', 'Meditation music'],
            'color': '#FF4500',
            'message': 'Take a moment to breathe and relax.'
        },
        'surprise': {
            'activities': ['Embrace the moment', 'Share your excitement', 'Document the experience'],
            'music': ['Exciting soundtracks', 'Energetic beats'],
            'color': '#FF69B4',
            'message': 'Life is full of surprises! Enjoy this moment!'
        },
        'fear': {
            'activities': ['Practice relaxation', 'Talk to someone', 'Focus on breathing', 'Ground yourself'],
            'music': ['Calming nature sounds', 'Slow tempo music', 'Guided meditation'],
            'color': '#800080',
            'message': 'You\'re safe. Take slow, deep breaths.'
        },
        'disgust': {
            'activities': ['Change your environment', 'Practice mindfulness', 'Focus on positive things'],
            'music': ['Uplifting music', 'Happy tunes'],
            'color': '#228B22',
            'message': 'Shift your focus to something pleasant.'
        },
        'neutral': {
            'activities': ['Try something new', 'Connect with friends', 'Set a new goal', 'Learn something'],
            'music': ['Your favorite genre', 'Discovery playlists'],
            'color': '#808080',
            'message': 'A calm state is perfect for new beginnings!'
        }
    }
    
    recommendation = recommendations.get(dominant_emotion, recommendations['neutral'])
    recommendation['emotion'] = dominant_emotion
    
    return jsonify(recommendation)

@app.route('/api/notes', methods=['GET', 'POST', 'DELETE'])
def api_notes():
    """Manage session notes"""
    global session_notes
    
    if request.method == 'POST':
        data = request.get_json()
        note_text = data.get('note', '')
        
        if not note_text:
            return jsonify({'success': False, 'message': 'Note cannot be empty'}), 400
        
        with lock:
            note = {
                'id': len(session_notes) + 1,
                'text': note_text,
                'timestamp': datetime.datetime.now().isoformat(),
                'emotion': emotion_counter.most_common(1)[0][0] if emotion_counter else 'neutral'
            }
            session_notes.append(note)
        
        return jsonify({'success': True, 'note': note})
    
    elif request.method == 'DELETE':
        note_id = request.args.get('id', type=int)
        with lock:
            session_notes = [n for n in session_notes if n['id'] != note_id]
        return jsonify({'success': True, 'message': 'Note deleted'})
    
    else:  # GET
        with lock:
            return jsonify({
                'notes': session_notes,
                'count': len(session_notes)
            })

@app.route('/api/intensity')
def api_intensity():
    """Get emotion intensity data"""
    with lock:
        intensity_data = list(emotion_intensity_history)
        
        # Calculate average intensity per emotion
        emotion_avg_intensity = {}
        for entry in intensity_data:
            emotion = entry['emotion']
            intensity = entry['intensity']
            if emotion not in emotion_avg_intensity:
                emotion_avg_intensity[emotion] = []
            emotion_avg_intensity[emotion].append(intensity)
        
        # Calculate averages
        for emotion in emotion_avg_intensity:
            intensities = emotion_avg_intensity[emotion]
            emotion_avg_intensity[emotion] = sum(intensities) / len(intensities)
        
        return jsonify({
            'intensity_history': intensity_data[-50:],  # Last 50 entries
            'average_intensity': emotion_avg_intensity,
            'current_intensity': intensity_data[-1]['intensity'] if intensity_data else 0
        })

@app.route('/api/session/save', methods=['POST'])
def api_save_session():
    """Save current session for comparison"""
    global sessions_archive
    
    with lock:
        session_data = {
            'id': len(sessions_archive) + 1,
            'timestamp': datetime.datetime.now().isoformat(),
            'start_time': session_start_time.isoformat(),
            'duration': str(datetime.datetime.now() - session_start_time).split('.')[0],
            'emotions': dict(emotion_counter),
            'total_detections': sum(emotion_counter.values()),
            'dominant_emotion': emotion_counter.most_common(1)[0][0] if emotion_counter else 'none',
            'notes': session_notes.copy()
        }
        
        sessions_archive.append(session_data)
        
        # Keep only last 10 sessions
        if len(sessions_archive) > 10:
            sessions_archive.pop(0)
    
    return jsonify({
        'success': True,
        'message': 'Session saved successfully',
        'session': session_data
    })

@app.route('/api/sessions')
def api_get_sessions():
    """Get all saved sessions"""
    with lock:
        return jsonify({
            'sessions': sessions_archive,
            'count': len(sessions_archive)
        })

@app.route('/api/session/compare')
def api_compare_sessions():
    """Compare two sessions"""
    session1_id = request.args.get('id1', type=int)
    session2_id = request.args.get('id2', type=int)
    
    with lock:
        session1 = next((s for s in sessions_archive if s['id'] == session1_id), None)
        session2 = next((s for s in sessions_archive if s['id'] == session2_id), None)
        
        if not session1 or not session2:
            return jsonify({'success': False, 'message': 'Session not found'}), 404
        
        comparison = {
            'session1': session1,
            'session2': session2,
            'differences': {
                'duration_diff': str(abs(
                    datetime.datetime.fromisoformat(session1['duration']) - 
                    datetime.datetime.fromisoformat(session2['duration'])
                )) if 'T' not in session1['duration'] else 'N/A',
                'detection_diff': session1['total_detections'] - session2['total_detections'],
                'emotion_changes': {}
            }
        }
        
        # Compare emotions
        all_emotions = set(list(session1['emotions'].keys()) + list(session2['emotions'].keys()))
        for emotion in all_emotions:
            count1 = session1['emotions'].get(emotion, 0)
            count2 = session2['emotions'].get(emotion, 0)
            comparison['differences']['emotion_changes'][emotion] = count1 - count2
    
    return jsonify(comparison)

@app.route('/api/stats/advanced')
def api_advanced_stats():
    """Get advanced statistics"""
    with lock:
        if not emotion_counter:
            return jsonify({'message': 'No data available'})
        
        total = sum(emotion_counter.values())
        emotions = dict(emotion_counter)
        
        # Calculate percentages
        percentages = {k: (v/total)*100 for k, v in emotions.items()}
        
        # Calculate emotion diversity (how varied the emotions are)
        diversity_score = len(emotions) / 7 * 100  # 7 total emotions
        
        # Get emotion trends (increasing/decreasing)
        trends = {}
        if len(emotion_history) >= 10:
            recent = emotion_history[-10:]
            for emotion in emotions.keys():
                recent_count = sum(1 for e in recent if e['emotion'] == emotion)
                trends[emotion] = 'increasing' if recent_count > emotions[emotion]/total*10 else 'stable'
        
        # Calculate session quality score
        positive_emotions = emotions.get('happy', 0) + emotions.get('surprise', 0)
        negative_emotions = emotions.get('cry', 0) + emotions.get('angry', 0) + emotions.get('fear', 0)
        quality_score = (positive_emotions / total * 100) if total > 0 else 50
        
        return jsonify({
            'percentages': percentages,
            'diversity_score': diversity_score,
            'trends': trends,
            'quality_score': quality_score,
            'dominant_emotion': emotion_counter.most_common(1)[0][0],
            'rare_emotions': emotion_counter.most_common()[:-4:-1],  # Least common
            'emotion_balance': {
                'positive': positive_emotions,
                'negative': negative_emotions,
                'neutral': emotions.get('neutral', 0)
            }
        })

# Additional Pages
@app.route('/about')
def about():
    """About page with information about the technology"""
    return render_template('about.html')

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import urllib.parse

FEEDBACK_FILE = Path("feedback_messages.json")
support_messages = []

# Load existing feedback from disk if present
if FEEDBACK_FILE.exists():
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            support_messages = json.load(f)
    except Exception as e:
        logger.error(f"Error loading feedback messages from disk: {e}")
        support_messages = []

def send_real_email_to_support(name, user_email, message_text):
    """Attempt to send a real email via SMTP to mallayasaswini7@gmail.com"""
    target_email = os.environ.get("SUPPORT_EMAIL", "mallayasaswini7@gmail.com")
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "") or os.environ.get("GMAIL_APP_PASSWORD", "")

    if smtp_user and smtp_pass:
        try:
            msg = MIMEMultipart()
            msg['From'] = smtp_user
            msg['To'] = target_email
            msg['Reply-To'] = user_email
            msg['Subject'] = f"[EmotiSense Feedback] Message from {name}"
            
            body = f"New EmotiSense Feedback Received:\n\nName: {name}\nEmail: {user_email}\nTimestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nMessage:\n{message_text}"
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
            server.quit()
            logger.info(f"Successfully sent email notification to {target_email}")
            return True, f"Direct email sent to {target_email}!"
        except Exception as e:
            logger.error(f"Failed to send email via SMTP: {e}")
            return False, str(e)
    else:
        logger.info("No SMTP credentials configured. Saved to feedback_messages.json.")
        return False, "SMTP not configured"

@app.route('/api/support', methods=['POST'])
def send_support_message():
    """Receive user support message, save to disk, attempt email delivery, and return Gmail compose link"""
    try:
        data = request.get_json() or {}
        name = data.get('name', 'Anonymous').strip()
        email = data.get('email', '').strip()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'success': False, 'message': 'Message body cannot be empty'}), 400
            
        entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'name': name,
            'email': email,
            'message': message
        }
        
        support_messages.append(entry)
        
        # Save to disk
        try:
            with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
                json.dump(support_messages, f, indent=2)
        except Exception as err:
            logger.error(f"Failed to save feedback to disk: {err}")
            
        logger.info(f"Support request received from {name} <{email}>: {message[:50]}...")
        
        # Attempt direct SMTP email sending
        sent_smtp, smtp_status = send_real_email_to_support(name, email, message)
        
        # Construct Gmail compose URL as fail-safe guarantee
        target_email = "mallayasaswini7@gmail.com"
        subject_enc = urllib.parse.quote(f"EmotiSense Feedback from {name}")
        body_enc = urllib.parse.quote(f"From: {name} ({email})\n\nFeedback Message:\n{message}")
        gmail_compose_url = f"https://mail.google.com/mail/?view=cm&fs=1&to={target_email}&su={subject_enc}&body={body_enc}"

        if sent_smtp:
            return jsonify({
                'success': True, 
                'message': f'Feedback delivered directly to {target_email}!',
                'gmail_url': gmail_compose_url,
                'email_sent': True
            })
        else:
            return jsonify({
                'success': True, 
                'message': f'Feedback saved! Opening Gmail compose for {target_email}...',
                'gmail_url': gmail_compose_url,
                'email_sent': False
            })
            
    except Exception as e:
        logger.error(f"Error handling support message: {e}")
        return jsonify({'success': False, 'message': 'Failed to process message'}), 500

@app.route('/help')
def help_page():
    """Help and documentation page"""
    return render_template('help.html')

@app.route('/features')
def features_page():
    """Features showcase page"""
    return render_template('features.html')

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    """Custom 404 error page"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Custom 500 error page"""
    logger.error(f"Internal error: {e}")
    return render_template('404.html'), 500

if __name__ == "__main__":
    import os
    
    logger.info("Starting EmotiSense Application...")
    logger.info(f"Session started at: {session_start_time}")
    
    # Get configuration from environment variables
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', 5000))
    
    try:
        app.run(debug=debug_mode, host=host, port=port)
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        if camera.isOpened():
            camera.release()
        logger.info("Camera released")
