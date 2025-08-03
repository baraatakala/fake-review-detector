"""
Quick fix script to ensure models are created and Flask app works
"""

import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def quick_train():
    """Quick training function"""
    print("🔧 Quick Fix: Creating models...")
    
    # Create sample data
    data = {
        'review_text': [
            "This product exceeded my expectations. Great quality and fast shipping.",
            "I've been using this for 3 months now and it's holding up well.",
            "Good value for money. Does exactly what it's supposed to do.",
            "Delivery was quick and the product matches the description perfectly.",
            "Solid build quality. My family loves it and we use it daily.",
            "Best product ever! Amazing! Buy it now! 5 stars!",
            "Incredible quality! Perfect! Amazing seller! Fast shipping!",
            "Outstanding! Excellent! Perfect! Amazing! Best purchase ever!",
            "WOW! Perfect product! Amazing quality! Super fast delivery!",
            "Best ever! Perfect! Amazing! Outstanding! Highly recommend!"
        ],
        'label': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0=genuine, 1=fake
    }
    
    df = pd.DataFrame(data)
    
    # Simple preprocessing
    df['processed_text'] = df['review_text'].str.lower().str.replace(r'[^a-zA-Z\s]', '', regex=True)
    
    # Prepare data
    X = df['processed_text']
    y = df['label']
    
    # Create and train vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(X)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_tfidf, y)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model and vectorizer
    with open('models/logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("✅ Models created successfully!")
    
    # Test prediction
    test_text = "This product is amazing! Perfect! Best ever!"
    test_processed = test_text.lower()
    test_tfidf = vectorizer.transform([test_processed])
    prediction = model.predict(test_tfidf)[0]
    probabilities = model.predict_proba(test_tfidf)[0]
    
    result = "FAKE" if prediction == 1 else "GENUINE"
    confidence = max(probabilities) * 100
    
    print(f"Test prediction: {result} ({confidence:.1f}% confidence)")
    print("Models are ready! You can now run: python app.py")

if __name__ == "__main__":
    quick_train()
