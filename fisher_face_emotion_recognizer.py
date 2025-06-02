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

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class FisherFaceEmotionRecognizer:
    """
    Implementation of the FisherFace algorithm for emotion detection
    as mentioned in the project documentation.
    
    This class provides methods for training and detecting emotions using
    the FisherFace approach, which combines PCA and LDA for optimal feature extraction.
    """
    
    def __init__(self, emotions):
        """
        Initialize the FisherFace emotion detector.
        """
        self.emotions = emotions
        self.pca = PCA(n_components=50)
        self.lda = LDA()
        self.trained = False
    
    def train(self, training_data, training_labels):
        """Train the emotion recognizer"""
        if len(training_data) == 0:
            print("No training data provided")
            return self
            
        # Flatten the images
        X = np.array([img.flatten() for img in training_data])
        y = np.array(training_labels)
        
        # Apply PCA for dimensionality reduction
        X_pca = self.pca.fit_transform(X)
        
        # Apply LDA for class separation
        self.lda.fit(X_pca, y)
        
        self.trained = True
        return self
    
    def predict(self, face_image):
        """Predict emotion from face image"""
        if not self.trained:
            # Return default values if not trained
            return 2, 0.5  # Neutral with medium confidence
        
        try:
            # Flatten the image
            X = face_image.flatten().reshape(1, -1)
            
            # Apply PCA transformation
            X_pca = self.pca.transform(X)
            
            # Predict using LDA
            label = self.lda.predict(X_pca)[0]
            
            # Calculate confidence (decision function gives distance to decision boundary)
            # Convert to a probability-like value between 0 and 1
            # Calculate confidence (decision function gives distance to decision boundary)
            # Convert to a probability-like value between 0 and 1
            proba = self.lda.predict_proba(X_pca)[0]
            confidence = proba[label]
            
            return label, confidence
        except Exception as e:
            print(f"Error in prediction: {e}")
            return 2, 0.5  # Default to neutral with medium confidence
