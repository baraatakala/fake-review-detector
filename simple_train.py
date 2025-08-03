#!/usr/bin/env python3
"""
Simple test script to train the model manually
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("=== Testing Model Training ===")
    
    # Load sample data
    print("Loading sample data...")
    try:
        df = pd.read_csv('sample_reviews.csv')
        print(f"Loaded {len(df)} reviews")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Simple preprocessing
    print("\nPreprocessing text...")
    df['processed_text'] = df['review_text'].str.lower().str.replace(r'[^a-zA-Z\s]', '', regex=True)
    
    # Prepare data
    X = df['processed_text']
    y = df['label']
    
    print(f"Sample texts:")
    for i in range(min(3, len(X))):
        print(f"  {i+1}. {X.iloc[i][:50]}... (Label: {y.iloc[i]})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Vectorize
    print("Creating TF-IDF vectors...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"TF-IDF shape: {X_train_tfidf.shape}")
    
    # Train model
    print("Training Logistic Regression...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Save model
    print("Saving model...")
    os.makedirs('models', exist_ok=True)
    
    with open('models/logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("✅ Model and vectorizer saved successfully!")
    
    # Test prediction
    test_text = "This product is amazing! Perfect! Best ever!"
    test_processed = test_text.lower().replace('[^a-zA-Z\\s]', '')
    test_tfidf = vectorizer.transform([test_processed])
    prediction = model.predict(test_tfidf)[0]
    probabilities = model.predict_proba(test_tfidf)[0]
    
    print(f"\nTest prediction:")
    print(f"Text: '{test_text}'")
    print(f"Prediction: {'Fake' if prediction == 1 else 'Genuine'}")
    print(f"Confidence: {max(probabilities):.4f}")

if __name__ == "__main__":
    main()
