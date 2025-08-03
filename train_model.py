"""
Fake Review Detector Model Training Script
This script trains a machine learning model to detect fake product reviews.
"""

import pandas as pd
import numpy as np
import pickle
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("Warning: Could not download NLTK data. Continuing without stopwords removal.")

class FakeReviewDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.label_encoder = None
        
    def preprocess_text(self, text):
        """
        Preprocess the input text for training and prediction
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords (optional, depends on NLTK availability)
        try:
            stop_words = set(stopwords.words('english'))
            word_tokens = word_tokenize(text)
            text = ' '.join([word for word in word_tokens if word not in stop_words])
        except:
            # If NLTK stopwords are not available, continue without stopword removal
            pass
        
        return text
    
    def create_sample_data(self, filename='sample_reviews.csv'):
        """
        Create a sample dataset for demonstration purposes
        In a real scenario, you would load your actual labeled dataset
        """
        sample_data = {
            'review_text': [
                # Genuine reviews
                "This product exceeded my expectations. Great quality and fast shipping.",
                "I've been using this for 3 months now and it's holding up well. Recommended.",
                "Good value for money. Does exactly what it's supposed to do.",
                "Delivery was quick and the product matches the description perfectly.",
                "Solid build quality. My family loves it and we use it daily.",
                "Great customer service when I had a question about installation.",
                "Works as advertised. No complaints so far after 2 weeks of use.",
                "The design is sleek and it fits perfectly in my kitchen.",
                "Assembly was straightforward with clear instructions included.",
                "Worth every penny. I would definitely buy this again.",
                
                # Fake reviews
                "Best product ever! Amazing! Buy it now! 5 stars!",
                "Incredible quality! Perfect! Amazing seller! Fast shipping! Recommended!",
                "Outstanding! Excellent! Perfect! Amazing! Best purchase ever!",
                "WOW! Perfect product! Amazing quality! Super fast delivery!",
                "Best ever! Perfect! Amazing! Outstanding! Highly recommend!",
                "Excellent! Perfect! Amazing quality! Fast shipping! Great seller!",
                "Perfect product! Amazing! Best quality! Super recommended!",
                "Outstanding quality! Perfect! Amazing! Best seller ever!",
                "Perfect! Amazing! Excellent! Best product! Fast delivery!",
                "Amazing! Perfect! Outstanding! Best quality! Highly recommend!"
            ],
            'label': [
                # 0 = Genuine, 1 = Fake
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Genuine reviews
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1   # Fake reviews
            ]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(filename, index=False)
        print(f"Sample dataset created: {filename}")
        return df
    
    def load_data(self, filename='sample_reviews.csv'):
        """
        Load the dataset from CSV file
        Expected columns: 'review_text', 'label'
        """
        if not os.path.exists(filename):
            print(f"Dataset file {filename} not found. Creating sample data...")
            return self.create_sample_data(filename)
        
        try:
            df = pd.read_csv(filename)
            print(f"Dataset loaded successfully: {len(df)} reviews")
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating sample data instead...")
            return self.create_sample_data(filename)
    
    def train_model(self, data_file='sample_reviews.csv'):
        """
        Train the fake review detection model
        """
        print("Loading and preprocessing data...")
        
        # Load data
        df = self.load_data(data_file)
        
        # Check if required columns exist
        if 'review_text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset must contain 'review_text' and 'label' columns")
        
        # Remove any missing values
        df = df.dropna(subset=['review_text', 'label'])
        
        # Preprocess the text
        print("Preprocessing text data...")
        df['processed_text'] = df['review_text'].apply(self.preprocess_text)
        
        # Remove empty texts after preprocessing
        df = df[df['processed_text'].str.len() > 0]
        
        print(f"Final dataset size: {len(df)} reviews")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        # Prepare features and labels
        X = df['processed_text']
        y = df['label']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Create and fit TF-IDF vectorizer
        print("Creating TF-IDF vectors...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,  # Limit to top 5000 features
            min_df=2,           # Ignore terms that appear in less than 2 documents
            max_df=0.8,         # Ignore terms that appear in more than 80% of documents
            ngram_range=(1, 2), # Use both unigrams and bigrams
            stop_words='english'
        )
        
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        print(f"TF-IDF matrix shape: {X_train_vectorized.shape}")
        
        # Train Logistic Regression model
        print("Training Logistic Regression model...")
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.model.fit(X_train_vectorized, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_vectorized)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Genuine', 'Fake']))
        
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy
    
    def save_model(self, model_dir='models'):
        """
        Save the trained model and vectorizer
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model must be trained before saving")
        
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(model_dir, 'logistic_regression_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save the vectorizer
        vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"Model saved to: {model_path}")
        print(f"Vectorizer saved to: {vectorizer_path}")
    
    def load_model(self, model_dir='models'):
        """
        Load the trained model and vectorizer
        """
        try:
            # Load the model
            model_path = os.path.join(model_dir, 'logistic_regression_model.pkl')
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load the vectorizer
            vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            print("Model and vectorizer loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, text):
        """
        Predict if a review is fake or genuine
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model must be loaded before making predictions")
        
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Vectorize the text
        text_vectorized = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.model.predict(text_vectorized)[0]
        probability = self.model.predict_proba(text_vectorized)[0]
        
        return {
            'prediction': 'Fake' if prediction == 1 else 'Genuine',
            'confidence': max(probability),
            'fake_probability': probability[1],
            'genuine_probability': probability[0]
        }

def main():
    """
    Main function to train and save the model
    """
    print("=== Fake Review Detector Model Training ===")
    
    # Initialize the detector
    detector = FakeReviewDetector()
    
    # Train the model
    try:
        accuracy = detector.train_model('sample_reviews.csv')
        
        # Save the model
        detector.save_model()
        
        print(f"\n=== Training Complete ===")
        print(f"Model accuracy: {accuracy:.4f}")
        print("Model and vectorizer saved successfully!")
        
        # Test the model with a sample prediction
        print(f"\n=== Testing Model ===")
        test_review = "This product is amazing! Perfect! Best ever! 5 stars!"
        result = detector.predict(test_review)
        print(f"Test review: '{test_review}'")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main()
