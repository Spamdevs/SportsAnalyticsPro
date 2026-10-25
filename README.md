
# SportsAnalytics Pro - Player Tracking and Analysis for Football Videos

An end-to-end computer vision web application for football player detection, tracking, team classification, and activity analysis.

## Features
- **Player Detection:** YOLOv8 (Ultralytics)
- **Player Tracking:** DeepSORT
- **Team Classification:** K-Means (HSV Color Space)
- **Activity Recognition:** MediaPipe Pose Estimation
- **Web Dashboard:** Flask + MySQL/SQLite

## Setup Instructions
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the AI pipeline on a video file using `ai_pipeline.py`.
4. Run the web application: `python app.py`
5. Access the dashboard at `http://localhost:5000`

## Results
The system successfully tracks players in broadcast footage, calculates real-world distances and speeds, and classifies activities (e.g., SPRINTING, JOGGING).

## Author
Ugwumadu Divine - University of Port Harcourt
