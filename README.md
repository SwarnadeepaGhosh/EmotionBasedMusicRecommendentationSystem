# Emotion-Based Music Recommendation System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-brightgreen)

Emotion-Based Music Recommendation System is an intelligent music player that automatically detects your facial expressions and plays music that matches your current emotional state. Using computer vision and machine learning techniques, the application captures your facial expressions through a webcam, identifies your emotion, and recommends music accordingly.


## Introduction
- Human emotion play a vital role nowadays
- Emotion expresses the human’s individual behaviour which can be of different forms.
- Extraction of the emotion states human’s individual state of behaviour.
- We proposed a system to arrange different music in different  categories such as happy, sad, angry etc.
- It’s a music player with chrome as front end has capability of detect emotion

## Objective
- To extract feature human face and detect emotion.
- To play music according to the emotion detected.
- Many existing techniques use previous data to suggest music and the other algorithms used are normally slow, usually they are less accurate and it even require additional hardware like EEG or physiological sensors.
- Facial expressions are captured a local capturing device or an inbuilt camera.
- We use algorithm for the recognition of the feature from the captured image.
- The proposed algorithm is based on the facial expression captured and music will be played automatically.

## LITERATURE SURVEY
- The process of multidimentional reduction by taking the primary data that is lowered to many other classes for sorting out or organizing.
- Emotion of user is extracted by capturing image through webcam
- Captured image is enhanced by the process of dimentional reduction
- These data converted in binary image format
- Face detected using Fisher Face and Harcascade methods
- The initial or the primary data taken from the human face that is loered to many other classes.
- These classes are sorted and organized using the above methods.
- Emotion isdetected by extracting the features from human face
- The main aim in feature extracting module is to diminish the number of resources required from the large sets of data.
- Features in an image consists 3 parts
 1. Boundaries/edges
 2. corners / projection points
 3. field points

## FISHER FACE ALGORITHM

- **PCA:** This image processing system is used for reducing the face space dimension using the principal component analysis (PCA) method           
- **FLD:** Then it applies fishers linear discriminant (FLD) or LDA method to obtain the feature of the image characteristics  
- we especially use this because it maximizes the separation between the classes in the training process 
- This algorithm helps to process the image recognition is done in Fisher face
- For matching faces algorithm we use minimum inclined it helps to classify the expression that implies the emotion of the user

## Features

- **Real-time Emotion Detection**: Analyzes facial expressions to detect five emotions (angry, happy, neutral, sad, surprised)
- **Personalized Music Selection**: Plays music that matches your detected emotional state
- **Interactive GUI**: Clean and intuitive interface with real-time emotion tracking
- **Emotion Stability Tracking**: Uses an emotion history window to prevent rapid song changes
- **Visualization**: Displays emotion histogram to track emotion patterns over time
- **Test Audio Generation**: Automatically creates test audio files if music is not available

## Technology Stack

- **Computer Vision**: OpenCV for face detection using Haar Cascade classifiers
- **Machine Learning**: Fisher Face algorithm (PCA + LDA) for emotion classification
- **Audio**: Pygame for music playback
- **GUI**: Tkinter and Matplotlib for the user interface
- **Data Processing**: NumPy and scikit-learn for data manipulation and machine learning

## Project Structure

```
EmotionBasedMusicRecommendentationSystem/
├── main.py                               # Main script to run
├── emotion_music_recommender.py          # Recommends musics based on emotion
├── fisher_face_emotion_recognizer.py     # Includes fisher face emotionalgorithm
├── haarcascade_frontalface_default.xml   # Haar cascade classifier for face detection
├── emotion_data/                         # Training data for emotion recognition
│   ├── angry/                            # Images of angry faces
│   ├── happy/                            # Images of happy faces
│   ├── neutral/                          # Images of neutral faces
│   ├── sad/                              # Images of sad faces
│   └── surprised/                        # Images of surprised faces
└── music/                                # Music files organized by emotion
    ├── angry/                            # Music for angry emotions
    ├── happy/                            # Music for happy emotions
    ├── neutral/                          # Music for neutral emotions
    ├── sad/                              # Music for sad emotions
    └── surprised/                        # Music for surprised emotions
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam

### Dependencies

Install the required packages:

```sh
pip install opencv-contrib-python numpy pygame scikit-learn pillow matplotlib
```

### Setup

1. Clone the repository:

```sh
   git clone https://github.com/SwarnadeepaGhosh/EmotionBasedMusicRecommendentationSystem.git
   cd EmotionBasedMusicRecommendentationSystem
```

2. Ensure the Haar cascade file is available:
   - The `haarcascade_frontalface_default.xml` file should be in the project directory
   - If missing, you can download it from OpenCV's GitHub repository

3. Prepare your training data (optional):
   - Create an `emotion_data` directory with subdirectories for each emotion
   - Add facial expression images to their respective emotion folders
   - If no training data is provided, the system will use default settings

4. Add your music (optional):
   - Place music files in the appropriate emotion folder under `music/`
   - Supported formats: `.wav`, `.mp3`, `.ogg`
   - If no music files are found, the system will create test audio files

## Usage

Run the application:

```sh
python main.py
```

### Using the Application

1. **Start Camera**: Click the "Start Camera" button to activate your webcam
2. **Play/Pause**: Toggle music playback
3. **Next Song**: Skip to another song for the current emotion
4. **Quit**: Close the application

The application will:
1. Detect your face in the webcam feed
2. Identify your emotional state
3. Play music that matches your emotion
4. Show a visualization of your emotion history

## Components

### EmotionMusicRecommender

The main class that coordinates the application:
- Manages the webcam feed and GUI
- Processes face detection and emotion recognition
- Controls the music playback based on detected emotions

### FisherFaceEmotionRecognizer

Implements emotion recognition using Fisher Face algorithm:
- Uses Principal Component Analysis (PCA) for dimensionality reduction
- Applies Linear Discriminant Analysis (LDA) for emotion classification
- Calculates confidence scores for detected emotions

## Customization

### Adding Custom Music

1. Place your music files in the corresponding emotion directories:
   ```
   music/angry/
   music/happy/
   music/neutral/
   music/sad/
   music/surprised/
   ```
2. Supported formats: `.wav`, `.mp3`, `.ogg`

### Training with Custom Data

1. Prepare grayscale facial images (ideally 48x48 pixels)
2. Organize images into emotion folders in the `emotion_data` directory
3. The system will automatically load and train on these images

### Tuning Parameters

Key parameters that can be adjusted in the code:
- `confidence_threshold`: Minimum confidence level to accept emotion detection (default: 0.6)
- `history_window`: Number of frames to consider for emotion stability (default: 10)
- `min_song_duration`: Minimum seconds between song changes (default: 5)

## Troubleshooting

### Camera Not Detected
- Ensure your webcam is properly connected
- Check that no other application is using the webcam
- Verify webcam permissions for the application

### Music Not Playing
- Check if audio files exist in the emotion directories
- Ensure audio files are in supported formats (`.wav`, `.mp3`, `.ogg`)
- Verify that pygame is properly installed

### Poor Emotion Recognition
- Ensure good lighting conditions for better face detection
- Try training with more diverse facial expression data
- Adjust the confidence threshold parameter

## Future Enhancements

- Add support for more emotions (disgust, fear, etc.)
- Implement deep learning-based emotion recognition (CNN)
- Allow creation of personalized emotion-music mappings
- Integrate with music streaming services
- Add speech recognition for voice commands
- Implement facial recognition to support multiple users

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV and Haar Cascade classifiers for face detection
- Fisher Face algorithm for emotion recognition
- Pygame for audio functionality
- Contributors and testers who helped improve this system