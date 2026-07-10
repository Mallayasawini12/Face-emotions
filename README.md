# EmotiSense 🧠

**Real-Time Facial Emotion Detection System**

EmotiSense is an advanced AI-powered emotion recognition application that uses deep learning to analyze facial expressions in real-time through your webcam.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.12+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## ✨ Features

- 🎥 **Real-Time Detection**: Instant emotion recognition with minimal latency
- 😊 **7 Emotions**: Detects happy, cry, angry, surprise, fear, disgust, and neutral
- 📊 **Advanced Analytics**: Interactive charts and detailed emotion breakdowns
- 💾 **Data Export**: Export analytics to CSV and JSON formats
- 📈 **History Tracking**: Records emotion changes over time with timestamps
- 🔌 **RESTful API**: Access emotion data programmatically
- 🎨 **Modern UI/UX**: Professional, responsive design
- 📱 **Mobile Friendly**: Works on desktop, tablet, and mobile devices

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- Webcam
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

1. **Clone or download the repository**
   ```bash
   cd FF
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   Navigate to: `http://127.0.0.1:5000`

## 📖 Usage

### Live Feed
1. Visit the homepage to start the live emotion detection
2. Allow camera access when prompted
3. Position your face clearly in view
4. Watch real-time emotion detection with confidence scores

### Analytics Dashboard
1. Click "View Analytics Dashboard" from the live feed
2. View detailed statistics and charts
3. Export data in CSV or JSON format
4. Reset statistics to start a new session

### API Endpoints

#### Get Current Emotions
```
GET /api/emotions
```
Returns current emotion statistics and session information.

#### Get History
```
GET /api/history
```
Retrieve emotion history with timestamps (last 50 entries).

#### Reset Statistics
```
POST /api/reset
```
Reset all emotion statistics and start a new session.

#### Export Data
```
GET /api/export/json
GET /api/export/csv
```
Download emotion data in JSON or CSV format.

## 🛠️ Technology Stack

- **Backend**: Flask (Python web framework)
- **Computer Vision**: OpenCV
- **Deep Learning**: TensorFlow, DeepFace
- **Data Processing**: NumPy, Pandas
- **Visualization**: Chart.js, Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3, JavaScript
- **UI Framework**: Custom CSS with modern design patterns

## 📁 Project Structure

```
FF/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── test_imports.py        # Package verification script
├── static/
│   └── style.css         # Application styles
├── templates/
│   ├── index.html        # Live feed page
│   ├── dashboard.html    # Analytics dashboard
│   ├── about.html        # About page
│   ├── help.html         # Help & documentation
│   └── 404.html          # Error page
└── .venv/                # Virtual environment (not tracked)
```

## 🎯 Use Cases

- **Mental Health**: Monitor emotional wellbeing and mood patterns
- **Education**: Assess student engagement and learning experiences
- **Business**: Analyze customer satisfaction and feedback
- **User Experience**: Test product interfaces and user reactions
- **Research**: Collect emotion data for psychological studies

## 📊 Detected Emotions

| Emotion  | Description |
|----------|-------------|
| Happy    | Positive, joyful expressions |
| Cry (Sad)| Sadness, distress, crying |
| Angry    | Frustration, anger, aggression |
| Surprise | Shock, amazement, surprise |
| Fear     | Anxiety, fear, concern |
| Disgust  | Disapproval, disgust, distaste |
| Neutral  | Calm, neutral expressions |

## 🔧 Troubleshooting

### Camera not working
- Check browser permissions
- Ensure no other application is using the camera
- Try refreshing the page

### Low accuracy
- Improve lighting conditions
- Face the camera directly
- Ensure your entire face is visible
- Avoid extreme angles or distances

### Installation issues
- Ensure Python 3.12+ is installed
- Try upgrading pip: `python -m pip install --upgrade pip`
- Install packages individually if bulk install fails

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📧 Support

For support, email support@emotisense.com or visit the Help page in the application.

## 🙏 Acknowledgments

- DeepFace library for emotion detection models
- TensorFlow team for the deep learning framework
- OpenCV community for computer vision tools
- Flask framework for web development

## 🔄 Version History

- **v1.0.0** (February 2026)
  - Initial release
  - Real-time emotion detection
  - Analytics dashboard
  - Data export functionality
  - RESTful API
  - Modern UI/UX design



**EmotiSense** - Powered by AI & Deep Learning
