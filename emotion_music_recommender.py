#!/usr/bin/env python3
"""
Emotion-Based Music Recommendation System
========================================
Emotion detection module using facial expressions

Author: Swarnadeepa Ghosh
Roll No: 90/CSE 230012
Reg No: 5080014 0f 2023-24
Subject: CSE (M.tech)
"""

import cv2
import numpy as np
import os
import random
import pygame
import threading
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from fisher_face_emotion_recognizer import FisherFaceEmotionRecognizer

class EmotionMusicRecommender:
    def __init__(self):
        # Initialize emotion labels and confidence threshold
        self.emotions = ["angry", "happy", "neutral", "sad", "surprised"]
        self.emotion_colors = {
            "angry": "#FF5555",      # Red
            "happy": "#FFDD55",      # Yellow
            "neutral": "#AAAAAA",    # Gray
            "sad": "#5555FF",        # Blue
            "surprised": "#FF55FF"   # Pink
        }
        self.confidence_threshold = 0.6  # Minimum confidence to accept emotion detection
        
        # Path to the cascade file
        self.face_cascade_path = 'haarcascade_frontalface_default.xml'
        self.face_detector = cv2.CascadeClassifier(self.face_cascade_path)
        
        # Initialize emotion recognizer
        self.emotion_recognizer = FisherFaceEmotionRecognizer(self.emotions)
        
        # Music library organized by emotion
        self.music_library = {
            "angry": ["angry_song1.wav", "angry_song2.wav"],
            "happy": ["happy_song1.wav", "happy_song2.wav", "happy_song3.wav"],
            "neutral": ["neutral_song1.wav", "neutral_song2.wav", "neutral_song3.wav"],
            "sad": ["sad_song1.wav", "sad_song2.wav", "sad_song3.wav"],
            "surprised": ["surprised_song1.wav", "surprised_song2.wav", "surprised_song3.wav"]
        }
        
        # Current emotion and playing status
        self.current_emotion = "neutral"
        self.is_playing = False
        self.current_song = ""
        self.emotion_history = []  # Store recent emotions for stability
        self.history_window = 10   # Number of frames to consider for emotion stability
        self.emotion_histogram = {emotion: 0 for emotion in self.emotions}
        
        # Initialize pygame for music playback
        pygame.mixer.init()
        
        # Initialize webcam
        self.cap = None
        self.is_webcam_active = False
        self.webcam_thread = None
        
        # GUI elements
        self.root = None
        self.video_label = None
        self.emotion_label = None
        self.song_label = None
        self.confidence_meter = None
        self.histogram_canvas = None
        
        # Detected face coordinates for animation
        self.face_rect = None
        
        # Flag to track if initial music has been played
        self.initial_music_played = False
        
    def load_training_data(self, data_folder="emotion_data"):
        """Load training data from the emotion data folder"""
        if not os.path.exists(data_folder):
            print(f"Training data folder {data_folder} does not exist")
            # Create directory structure for demo
            for emotion in self.emotions:
                os.makedirs(os.path.join(data_folder, emotion), exist_ok=True)
            return False
        
        print("Loading training data...")
        training_data = []
        training_labels = []
        
        for idx, emotion in enumerate(self.emotions):
            emotion_folder = os.path.join(data_folder, emotion)
            if not os.path.exists(emotion_folder):
                continue
            
            files = [f for f in os.listdir(emotion_folder) 
                    if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            for image_file in files:
                image_path = os.path.join(emotion_folder, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    continue
                
                # Resize to a fixed size for consistency
                image = cv2.resize(image, (48, 48))
                training_data.append(image)
                training_labels.append(idx)
        
        if len(training_data) == 0:
            print("No training data found - using default classifier")
            return False
        
        # Train the emotion recognizer
        self.emotion_recognizer.train(training_data, training_labels)
        print(f"Trained with {len(training_data)} images")
        return True
    
    def detect_emotion(self, frame):
        """Detect faces and emotions in a frame"""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        detected_emotion = None
        confidence = 0
        face_found = False
        
        for (x, y, w, h) in faces:
            face_found = True
            self.face_rect = (x, y, w, h)  # Store face position for animation
            
            # Create fancy rectangle with rounded corners
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract and preprocess face
            face_roi = gray[y:y+h, x:x+w]
            try:
                face_roi = cv2.resize(face_roi, (48, 48))
                
                # Predict emotion
                emotion_idx, conf = self.emotion_recognizer.predict(face_roi)
                
                if 0 <= emotion_idx < len(self.emotions):
                    detected_emotion = self.emotions[emotion_idx]
                    confidence = conf
                    
                    # Get color for detected emotion
                    emotion_color = self.emotion_colors.get(detected_emotion, (0, 255, 0))
                    if isinstance(emotion_color, str):
                        # Convert hex to RGB
                        emotion_color = tuple(int(emotion_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                        emotion_color = emotion_color[::-1]  # RGB to BGR
                    
                    # Display emotion on frame with colored background
                    text_size = cv2.getTextSize(detected_emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                    cv2.rectangle(frame, (x, y-text_size[1]-10), (x+text_size[0]+10, y), emotion_color, -1)
                    cv2.putText(frame, detected_emotion, (x+5, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            except Exception as e:
                print(f"Error in emotion detection: {e}")
        
        if not face_found:
            self.face_rect = None
            
        return frame, detected_emotion, confidence
    
    def update_emotion_history(self, emotion, confidence):
        """Update emotion history for stability analysis"""
        if emotion is None:
            return self.current_emotion
        
        # Only consider emotions detected with sufficient confidence
        if confidence < self.confidence_threshold:
            return self.current_emotion
            
        # Update emotion history
        self.emotion_history.append(emotion)
        if len(self.emotion_history) > self.history_window:
            self.emotion_history.pop(0)
        
        # Count occurrences of each emotion in history
        emotion_counts = {}
        for e in self.emotion_history:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        
        # Update histogram for visualization
        for e in self.emotions:
            self.emotion_histogram[e] = emotion_counts.get(e, 0)
            
        # Only change current emotion if a new emotion is stable
        # (appears in more than half of recent frames)
        most_common_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else self.current_emotion
        if emotion_counts.get(most_common_emotion, 0) > self.history_window / 2:
            if most_common_emotion != self.current_emotion:
                print(f"Emotion changed: {self.current_emotion} -> {most_common_emotion}")
                return most_common_emotion
        
        return self.current_emotion
    
    def play_music_for_emotion(self, emotion):
        """Play a random song based on detected emotion"""
        if emotion not in self.music_library or len(self.music_library[emotion]) == 0:
            print(f"No songs available for emotion: {emotion}")
            return
        
        # Stop current song if playing
        if self.is_playing:
            pygame.mixer.music.stop()
        
        # Select a random song for the emotion
        song = random.choice(self.music_library[emotion])
        
        # Ensure the song file has an extension
        if not (song.endswith('.wav') or song.endswith('.mp3') or song.endswith('.ogg')):
            song += '.wav'
        
        # Create directory structure if needed
        emotion_dir = os.path.join("music", emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        
        song_path = os.path.join(emotion_dir, song)
        
        # Check if file exists or create test WAV file for demo
        if not os.path.exists(song_path):
            print(f"Song file not found: {song_path}, creating test audio...")
            song_path = self.create_test_audio_file(song_path)
            if not song_path:
                print("Failed to create test audio file")
                return
        
        try:
            # Load and play the music file
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play()
            print(f"Now playing: {song} for emotion: {emotion}")
            self.is_playing = True
            self.current_song = song
            
            # Update UI
            if self.song_label:
                self.song_label.config(text=f"🎵 Now Playing: {song}", 
                                     foreground=self.emotion_colors[emotion])
        except Exception as e:
            print(f"Error playing music: {e}")
            # Try to play a system sound as fallback
            try:
                print(f"Attempting to use system beep as fallback...")
                frequency = 440  # A4 note
                duration = 500   # ms
                
                # Generate sound buffer
                buf = self.generate_tone(frequency, duration/1000)
                
                # Create a temporary sound file
                temp_path = os.path.join("music", "temp_sound.wav")
                self.save_wav_file(temp_path, buf)
                
                # Play the temporary sound
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                
                self.is_playing = True
                self.current_song = "System beep (fallback)"  # Set a meaningful fallback name
                if self.song_label:
                    self.song_label.config(text=f"🎵 Now Playing: {self.current_song}", 
                                         foreground=self.emotion_colors[emotion])
            except Exception as e2:
                print(f"Fallback sound also failed: {e2}")
                self.is_playing = False
                self.current_song = ""
    
    def save_wav_file(self, filepath, audio_data):
        """Save audio data as WAV file using pygame's mixer"""
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                
            # Create a Sound object and save it
            sound = pygame.mixer.Sound(buffer=audio_data)
            sound.set_volume(0.5)  # Set volume to 50%
            pygame.mixer.sndarray.make_sound(audio_data).write(filepath)
            return filepath
        except Exception as e:
            print(f"Error saving WAV file: {e}")
            return None
                
    def create_test_audio_file(self, filepath):
        """Create a test audio file with a simple tone for demo purposes"""
        try:
            # Generate a simple WAV file
            sample_rate = 44100
            
            # Pick a frequency based on the file path (to make different emotions sound different)
            if "happy" in filepath.lower():
                frequency = 523.25  # C5 - higher, happier note
                duration = 1.0      # 1 second
            elif "sad" in filepath.lower():
                frequency = 261.63  # C4 - lower, sadder note
                duration = 1.5      # 1.5 seconds
            elif "angry" in filepath.lower():
                frequency = 349.23  # F4 - tense note
                duration = 0.7      # 0.7 seconds
            elif "surprised" in filepath.lower():
                frequency = 440.00  # A4 - surprise!
                duration = 0.5      # 0.5 seconds
            else:
                frequency = 392.00  # G4 - neutral
                duration = 1.0      # 1 second
            
            # Generate tone
            data = self.generate_tone(frequency, duration)
            
            # Ensure the filepath ends with .wav
            if not filepath.lower().endswith(('.wav', '.mp3', '.ogg')):
                filepath = filepath + '.wav'
            
            # Save the sound file
            self.save_wav_file(filepath, data)
            print(f"Created test audio file at {filepath}")
            
            # Update the song name in our library if needed
            song_name = os.path.basename(filepath)
            for e in self.music_library:
                for i, s in enumerate(self.music_library[e]):
                    if s == os.path.splitext(song_name)[0] or s + '.wav' == song_name:
                        self.music_library[e][i] = song_name
            
            return filepath
        except Exception as e:
            print(f"Failed to create test audio: {e}")
            return None
    
    def generate_tone(self, frequency, duration):
        """Generate a simple sine wave tone"""
        sample_rate = 44100
        n_samples = int(duration * sample_rate)
        
        # Generate sine wave with fade in/out to avoid clicks
        fade = int(sample_rate * 0.05)  # 50ms fade
        buf = np.zeros(n_samples, dtype=np.float32)
        
        # Generate main tone
        t = np.arange(n_samples) / sample_rate
        buf = np.sin(2 * np.pi * frequency * t)
        
        # Apply fade in/out
        if fade > 0:
            buf[:fade] *= np.linspace(0, 1, fade)
            buf[-fade:] *= np.linspace(1, 0, fade)
        
        # Convert to 16-bit signed integers
        buf = (buf * 32767).astype(np.int16)
        
        return buf
    
    def update_histogram(self):
        """Update emotion histogram visualization"""
        if not self.histogram_canvas:
            return
            
        # Clear previous plot
        self.histogram_fig.clear()
        ax = self.histogram_fig.add_subplot(111)
        
        # Plot histogram
        emotions = list(self.emotion_histogram.keys())
        counts = list(self.emotion_histogram.values())
        colors = [self.emotion_colors[e] for e in emotions]
        
        ax.bar(emotions, counts, color=colors)
        ax.set_ylabel('Frequency')
        ax.set_title('Emotion History')
        
        # Refresh canvas
        self.histogram_canvas.draw()
        
    def start_webcam(self):
        """Start webcam and emotion detection"""
        if self.is_webcam_active:
            return
            
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            if self.emotion_label:
                self.emotion_label.config(text="⚠️ Error: Could not open webcam")
            return
            
        self.is_webcam_active = True
        self.webcam_thread = threading.Thread(target=self.webcam_loop)
        self.webcam_thread.daemon = True
        self.webcam_thread.start()
        
        # Play initial music for default emotion after a short delay
        def play_initial_music():
            time.sleep(1)  # Short delay to let webcam initialize
            if not self.initial_music_played:
                self.play_music_for_emotion(self.current_emotion)
                self.initial_music_played = True
                
        initial_music_thread = threading.Thread(target=play_initial_music)
        initial_music_thread.daemon = True
        initial_music_thread.start()
        
    def stop_webcam(self):
        """Stop webcam"""
        self.is_webcam_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def webcam_loop(self):
        """Main webcam processing loop"""
        last_song_change = time.time()
        min_song_duration = 5  # Minimum seconds between song changes
        
        while self.is_webcam_active and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect emotion
            processed_frame, detected_emotion, confidence = self.detect_emotion(frame)
            
            # Update emotion stability tracking
            previous_emotion = self.current_emotion
            new_emotion = self.update_emotion_history(detected_emotion, confidence)
            
            # Check if emotion has changed
            emotion_changed = (new_emotion != previous_emotion)
            self.current_emotion = new_emotion
            
            # Update UI components
            self.update_ui(processed_frame, detected_emotion, confidence)
            
            # Play music if emotion changed (with minimum time between changes)
            current_time = time.time()
            if emotion_changed and (current_time - last_song_change) > min_song_duration:
                print(f"Emotion changed to {self.current_emotion}, playing corresponding music")
                self.play_music_for_emotion(self.current_emotion)
                last_song_change = current_time
            
            # Slight pause to reduce CPU usage
            time.sleep(0.03)
        
    def update_ui(self, frame, detected_emotion, confidence):
        """Update all UI components"""
        # Update video feed
        if self.video_label:
            cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image)
            
            # Resize if necessary
            if hasattr(self, 'video_size'):
                pil_image = pil_image.resize(self.video_size, Image.LANCZOS)
                
            tk_image = ImageTk.PhotoImage(image=pil_image)
            self.video_label.config(image=tk_image)
            self.video_label.image = tk_image
        
        # Update emotion label
        if self.emotion_label:
            emotion_color = self.emotion_colors.get(self.current_emotion, "#FFFFFF")
            self.emotion_label.config(
                text=f"😊 Current Emotion: {self.current_emotion.upper()}", 
                foreground=emotion_color
            )
        
        # Update confidence meter
        if self.confidence_meter and detected_emotion:
            self.confidence_meter['value'] = confidence * 100
        
        # Update histogram
        self.update_histogram()
            
    def create_gui(self):
        """Create an attractive GUI for the application"""
        self.root = tk.Tk()
        self.root.title("Mood Music - Emotion Based Music Player")
        self.root.geometry("950x800")  # Increased size to accommodate all elements
        self.root.configure(bg="#2E3440")  # Dark background
        
        # Create style for ttk widgets
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#2E3440")
        style.configure("TLabel", background="#2E3440", foreground="#ECEFF4", font=("Arial", 12))
        style.configure("TButton", background="#5E81AC", foreground="#ECEFF4", font=("Arial", 12))
        style.configure("Horizontal.TProgressbar", background="#88C0D0", troughcolor="#3B4252")
        
        # Main frame - Using grid layout for better control
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        main_frame.columnconfigure(0, weight=1)  # Make column expandable
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, sticky="ew", pady=10)
        
        title_label = ttk.Label(header_frame, text="Mood Music Player", font=("Arial", 20, "bold"))
        title_label.pack(side=tk.LEFT)
        
        # Split the main area into two panels side by side
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=1, column=0, sticky="nsew", pady=10)
        content_frame.columnconfigure(0, weight=2)  # Video panel gets more space
        content_frame.columnconfigure(1, weight=1)  # Info panel gets less space
        
        # Left panel for video
        video_frame = ttk.Frame(content_frame)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        self.video_size = (480, 360)  # Smaller video to fit alongside controls
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(pady=5)
        
        # Right panel for info and histogram
        info_panel = ttk.Frame(content_frame)
        info_panel.grid(row=0, column=1, sticky="nsew")
        
        # Emotion display
        self.emotion_label = ttk.Label(info_panel, text="😊 Current Emotion: Waiting...", 
                                     font=("Arial", 16, "bold"))
        self.emotion_label.pack(pady=5, fill=tk.X)
        
        # Song display
        self.song_label = ttk.Label(info_panel, text="🎵 Song: Not playing", 
                                   font=("Arial", 14))
        self.song_label.pack(pady=5, fill=tk.X)
        
        # Confidence meter
        meter_frame = ttk.Frame(info_panel)
        meter_frame.pack(fill=tk.X, pady=5)
        
        confidence_label = ttk.Label(meter_frame, text="Confidence: ")
        confidence_label.pack(side=tk.LEFT)
        
        self.confidence_meter = ttk.Progressbar(meter_frame, orient=tk.HORIZONTAL, 
                                             length=200, mode='determinate')
        self.confidence_meter.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Control buttons in info panel
        control_frame = ttk.Frame(info_panel)
        control_frame.pack(pady=10, fill=tk.X)
        
        start_button = ttk.Button(control_frame, text="Start Camera", command=self.start_webcam)
        start_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        play_button = ttk.Button(control_frame, text="Play/Pause", command=self.toggle_play)
        play_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        next_button = ttk.Button(control_frame, text="Next Song", command=self.next_song)
        next_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        quit_button = ttk.Button(control_frame, text="Quit", command=self.quit_app)
        quit_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Emotion histogram in bottom frame
        histogram_frame = ttk.Frame(main_frame)
        histogram_frame.grid(row=2, column=0, sticky="ew", pady=10)
        
        self.histogram_fig = plt.Figure(figsize=(9, 3), tight_layout=True)
        self.histogram_canvas = FigureCanvasTkAgg(self.histogram_fig, master=histogram_frame)
        self.histogram_canvas.get_tk_widget().pack(fill=tk.X, expand=True)
        
        # Set up clean exit
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
        
        return self.root
    
    def toggle_play(self):
        """Toggle play/pause of music"""
        if not self.current_song:
            # No song is selected, play one for current emotion
            self.play_music_for_emotion(self.current_emotion)
            return
            
        if self.is_playing:
            pygame.mixer.music.pause()
            print("Music paused")
            self.is_playing = False
            if self.song_label:
                # Make sure to show the current song name when paused
                self.song_label.config(text=f"🎵 Paused: {self.current_song}")
        else:
            pygame.mixer.music.unpause()
            print("Music resumed")
            self.is_playing = True
            if self.song_label:
                # Make sure to show the current song name when resumed
                self.song_label.config(text=f"🎵 Now Playing: {self.current_song}")
    
    def next_song(self):
        """Play next song for current emotion"""
        emotion_to_use = self.current_emotion
        print(f"Next song requested for emotion: {emotion_to_use}")
        
        # Check if music library has songs for this emotion
        if emotion_to_use in self.music_library and len(self.music_library[emotion_to_use]) > 0:
            self.play_music_for_emotion(emotion_to_use)
        else:
            print(f"No songs available for emotion: {emotion_to_use}, using neutral")
            self.play_music_for_emotion("neutral")
        
    def init_music_library(self):
        """Initialize music library with existing files or create test files"""
        base_dir = "music"
        os.makedirs(base_dir, exist_ok=True)
        
        # Scan for existing music files
        for emotion in self.emotions:
            emotion_dir = os.path.join(base_dir, emotion)
            os.makedirs(emotion_dir, exist_ok=True)
            
            # Check for existing music files
            music_files = [f for f in os.listdir(emotion_dir) 
                         if f.endswith(('.mp3', '.wav', '.ogg'))]
            
            # If no music files found, create test files
            if not music_files:
                print(f"No music files found for {emotion}, creating test files...")
                # Create 3 test files for each emotion
                for i in range(3):
                    song_name = f"{emotion}_song{i+1}.wav"
                    song_path = os.path.join(emotion_dir, song_name)
                    created_path = self.create_test_audio_file(song_path)
                    if created_path:
                        music_files.append(os.path.basename(created_path))
            
            # Update music library with found files
            if music_files:
                self.music_library[emotion] = music_files
                print(f"Music for {emotion}: {', '.join(music_files)}")
            else:
                # Ensure each emotion has at least one song
                default_song = f"{emotion}_default.wav"
                default_path = os.path.join(emotion_dir, default_song)
                created_path = self.create_test_audio_file(default_path)
                if created_path:
                    self.music_library[emotion] = [os.path.basename(created_path)]
    
    def quit_app(self):
        """Quit the application"""
        self.stop_webcam()
        if self.is_playing:
            pygame.mixer.music.stop()
        
        if self.root:
            self.root.quit()
            self.root.destroy()
    
    def run(self):
        """Run the emotion-based music recommender"""
        # Load and train data
        self.load_training_data()
        
        # Initialize music library
        self.init_music_library()
        
        # Create and run GUI
        root = self.create_gui()
        root.mainloop()