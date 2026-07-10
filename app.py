from flask import Flask, render_template, Response, jsonify, send_file, request
import cv2
import logging
try:
    from deepface import DeepFace
except Exception:
    DeepFace = None
    logger = logging.getLogger(__name__)
    logger.warning('deepface not available; emotion analysis will be disabled')
from collections import Counter, deque
import threading
import datetime
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

app = Flask(__name__)

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
    try:
        # allow numeric source (webcam index) or file path
        cam_index = int(source)
        cam = cv2.VideoCapture(cam_index)
    except Exception:
        cam = cv2.VideoCapture(source)

    if cam is None or not cam.isOpened():
        logger.warning(f"Camera source '{source}' not available; using placeholder frames.")
        return None

    return cam


camera = init_camera()

# Initialize face cascade and ONNX model for fallback/alternative emotion analysis
face_cascade = None
emotion_net = None

def init_onnx_model():
    global face_cascade, emotion_net
    # 1. Load Haar Cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if os.path.exists(cascade_path):
        face_cascade = cv2.CascadeClassifier(cascade_path)
    else:
        logger.error(f"Haar cascade XML not found at {cascade_path}")
        
    # 2. Download and load ONNX model
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

def generate_frames():
    global current_frame, camera
    scores_history = deque(maxlen=5)  # Rolling buffer for temporal smoothing (faster reaction time)
    while True:
        if camera is None:
            # Create a placeholder frame when no camera is available
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, 'No camera available', (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
            # sleep a short while to avoid tight-looping
            # (yielding frames at ~5 FPS)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue

        success, frame = camera.read()
        if not success:
            # if camera read fails, switch to placeholder next loop
            logger.warning('Camera read failed; switching to placeholder frames.')
            camera.release()
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
                
                # Map 'sad' to 'cry' for better user understanding
                display_emotion = 'cry' if emotion == 'sad' else emotion
                counter_emotion = 'cry' if emotion == 'sad' else emotion
                
                # Calculate emotion intensity (dominant emotion score)
                emotion_intensity = max(emotion_scores.values())

                with lock:
                    current_frame = frame.copy()  # Store current frame for snapshots
                    emotion_counter[counter_emotion] += 1
                    
                    # Track history with timestamp (limit to last 100 entries)
                    emotion_history.append({
                        'emotion': counter_emotion,
                        'timestamp': datetime.datetime.now().isoformat(),
                        'scores': emotion_scores
                    })
                    if len(emotion_history) > 100:
                        emotion_history.pop(0)
                    
                    # Track emotion intensity
                    emotion_intensity_history.append({
                        'timestamp': datetime.datetime.now().isoformat(),
                        'emotion': counter_emotion,
                        'intensity': emotion_intensity
                    })

                # Draw emotion and confidence
                cv2.putText(frame, f'Emotion: {display_emotion}', (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
                
                # Show top 3 emotions with scores
                y_offset = 80
                sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                for emo, score in sorted_emotions:
                    display_emo = 'cry' if emo == 'sad' else emo
                    cv2.putText(frame, f'{display_emo}: {score:.1f}%', (30, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, (255, 255, 0), 1)
                    y_offset += 30

            except Exception:
                # Draw error message
                cv2.putText(frame, 'No face detected', (30, 40),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 0, 255), 2)
        elif emotion_net is not None and face_cascade is not None:
            try:
                # Convert to grayscale for face detection and processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                if len(faces) > 0:
                    # Use the largest face
                    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                    
                    # Draw a rectangle around the detected face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Tighten crop slightly (5% padding) to focus on facial features
                    pad_w = int(w * 0.05)
                    pad_h = int(h * 0.05)
                    x1 = max(0, x + pad_w)
                    y1 = max(0, y + pad_h)
                    x2 = min(frame.shape[1], x + w - pad_w)
                    y2 = min(frame.shape[0], y + h - pad_h)
                    
                    # Crop and prepare the face ROI for the ONNX model
                    face_roi = gray[y1:y2, x1:x2]
                    face_roi = cv2.resize(face_roi, (64, 64))
                    
                    # Preprocess: (pixel_val - 127.5) / 127.5
                    normalized = (face_roi.astype(np.float32) - 127.5) / 127.5
                    blob = np.expand_dims(np.expand_dims(normalized, axis=0), axis=0)
                    
                    # Run inference
                    emotion_net.setInput(blob)
                    preds = emotion_net.forward()
                    
                    # The ONNX model already outputs softmax probabilities
                    scores = preds[0]
                    
                    # Map raw scores to the target 7 emotions
                    current_scores = {
                        'neutral': float(scores[0] + scores[7]) * 100,  # Combine neutral and contempt
                        'happy': float(scores[1]) * 100,
                        'surprise': float(scores[2]) * 100,
                        'sad': float(scores[3]) * 100,
                        'angry': float(scores[4]) * 100,
                        'disgust': float(scores[5]) * 100,
                        'fear': float(scores[6]) * 100
                    }
                    
                    # Smooth scores over the rolling window
                    scores_history.append(current_scores)
                    emotion_scores = {}
                    for key in current_scores.keys():
                        emotion_scores[key] = sum(s[key] for s in scores_history) / len(scores_history)
                    
                    emotion = max(emotion_scores, key=emotion_scores.get)
                    emotion_intensity = max(emotion_scores.values())
                    
                    # Keep 'cry' for internal tracking/database, but display 'sad' to the user
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
                    
                    # Annotate the frame with bounding box and emotion
                    cv2.putText(frame, f'Emotion: {display_emotion}', (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)
                    
                    # Show top 3 emotions with scores
                    y_offset = y + h + 25
                    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    for emo, score in sorted_emotions:
                        display_emo = 'sad' if emo in ['sad', 'cry'] else emo
                        cv2.putText(frame, f'{display_emo}: {score:.1f}%', (x, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   0.5, (255, 255, 0), 1)
                        y_offset += 20
                else:
                    scores_history.clear()  # Clear history buffer when no face is detected
                    cv2.putText(frame, 'No face detected', (30, 40),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 0, 255), 2)
            except Exception as e:
                logger.error(f"Error in ONNX fallback: {e}")
                cv2.putText(frame, 'Analysis Error', (30, 40),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 0, 255), 2)
        else:
            # Fallback if no models are available
            cv2.putText(frame, 'Emotion Engine Offline', (30, 40),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

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

# API Endpoints
@app.route('/api/emotions')
def api_emotions():
    """Get current emotion statistics"""
    with lock:
        data = {
            'emotions': dict(emotion_counter),
            'total_detections': sum(emotion_counter.values()),
            'session_start': session_start_time.isoformat(),
            'session_duration': str(datetime.datetime.now() - session_start_time).split('.')[0],
            'timestamp': datetime.datetime.now().isoformat()
        }
    return jsonify(data)

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
    global emotion_counter, emotion_history, session_start_time
    
    try:
        with lock:
            emotion_counter.clear()
            emotion_history.clear()
            session_start_time = datetime.datetime.now()
        
        logger.info("Statistics reset successfully")
        return jsonify({
            'success': True,
            'message': 'Statistics reset successfully',
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

@app.route('/api/snapshot', methods=['POST'])
def api_snapshot():
    """Capture current frame as snapshot"""
    global current_frame, saved_snapshots
    
    with lock:
        if current_frame is None:
            return jsonify({'success': False, 'message': 'No frame available'}), 400
        
        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', current_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Get current emotion if available
        current_emotion = emotion_counter.most_common(1)[0][0] if emotion_counter else 'unknown'
        
        snapshot_data = {
            'id': len(saved_snapshots) + 1,
            'timestamp': datetime.datetime.now().isoformat(),
            'emotion': current_emotion,
            'image': f'data:image/jpeg;base64,{img_base64}'
        }
        
        saved_snapshots.append(snapshot_data)
        
        # Keep only last 20 snapshots
        if len(saved_snapshots) > 20:
            saved_snapshots.pop(0)
    
    return jsonify({
        'success': True,
        'message': 'Snapshot captured',
        'snapshot': snapshot_data
    })

@app.route('/api/snapshots')
def api_get_snapshots():
    """Get all saved snapshots"""
    with lock:
        return jsonify({
            'snapshots': saved_snapshots,
            'count': len(saved_snapshots)
        })

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
