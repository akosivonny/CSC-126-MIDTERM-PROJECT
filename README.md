# People Counting System (Web-Based)

A beginner-friendly computer vision project using Python, Flask, HTML, and OpenCV. It accepts either a static image or a recorded video.

## Features
- Select static image or recorded video.
- Detect people with OpenCV's built-in HOG + SVM people detector.
- Draw green bounding boxes and labels.
- For images, show the number of detections in the image.
- For videos, show the maximum detections in a single frame and play the processed video.

## Setup (Windows)
1. Install Python 3.10 or newer.
2. Extract the ZIP and open this folder in VS Code.
3. In the terminal, optionally create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run:
   ```bash
   python app.py
   ```
6. Open `http://127.0.0.1:5000` in your browser.
7. Select Static image or Recorded video, choose a file, and click Detect People.

## Files
- `app.py`: Flask routes and OpenCV detection.
- `templates/index.html`: compact HTML/CSS web interface.
- `requirements.txt`: required Python libraries.

## OpenCV notes
This starter version uses image resizing/processing and the HOG + SVM detector. The video feature processes frames and adds bounding-box overlays. The detector can miss people or produce false detections, particularly in crowded scenes, side views, or images where people are partly hidden.

For the CSC 126 rubric's minimum of three laboratory topics, check the techniques you actually covered in class and add another clearly implemented topic (for example, motion detection or Haar Cascade facial detection). Do not claim a technique is included unless it is implemented.
