from flask import Flask, render_template, request, jsonify
import pickle
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import os
from datetime import datetime

app = Flask(__name__)

# Global variables to store the model and vectorizer
model = None
vectorizer = None

# Analytics storage
ANALYTICS_FILE = 'analytics.json'
analytics_data = {
    'total_predictions': 0,
    'fake_predictions': 0,
    'genuine_predictions': 0,
    'predictions_today': 0,
    'last_reset': datetime.now().strftime('%Y-%m-%d')
}

def load_analytics():
    """Load analytics data from file"""
    global analytics_data
    try:
        if os.path.exists(ANALYTICS_FILE):
            with open(ANALYTICS_FILE, 'r') as f:
                analytics_data = json.load(f)
        
        # Reset daily counter if new day
        today = datetime.now().strftime('%Y-%m-%d')
        if analytics_data.get('last_reset') != today:
            analytics_data['predictions_today'] = 0
            analytics_data['last_reset'] = today
            save_analytics()
    except Exception as e:
        print(f"Error loading analytics: {e}")

def save_analytics():
    """Save analytics data to file"""
    try:
        with open(ANALYTICS_FILE, 'w') as f:
            json.dump(analytics_data, f)
    except Exception as e:
        print(f"Error saving analytics: {e}")

def update_analytics(prediction):
    """Update analytics with new prediction"""
    global analytics_data
    analytics_data['total_predictions'] += 1
    analytics_data['predictions_today'] += 1
    
    if prediction == 1:
        analytics_data['fake_predictions'] += 1
    else:
        analytics_data['genuine_predictions'] += 1
    
    save_analytics()

def analyze_text_features(text):
    """Analyze additional text features"""
    if not text:
        return {}
    
    words = text.split()
    sentences = text.split('.')
    
    # Count exclamation marks and capital letters (indicators of fake reviews)
    exclamation_count = text.count('!')
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    
    # Count superlatives (often overused in fake reviews)
    superlatives = ['best', 'worst', 'amazing', 'terrible', 'perfect', 'awful', 'incredible', 'fantastic']
    superlative_count = sum(text.lower().count(word) for word in superlatives)
    
    return {
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'character_count': len(text),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'exclamation_count': exclamation_count,
        'caps_ratio': round(caps_ratio * 100, 2),
        'superlative_count': superlative_count
    }

def create_simple_model():
    """Create a simple model if the trained models are not available"""
    global model, vectorizer
    try:
        print("Creating simple fallback model...")
        
        # Sample training data (minimal for fallback)
        fake_reviews = [
            "Amazing! Best product ever! Perfect! 5 stars! Must buy!",
            "Incredible! Outstanding! Fantastic! Amazing quality! Perfect!",
            "Best ever! Amazing! Perfect! Incredible! Must have!",
            "Outstanding! Perfect! Amazing! Best quality! 5 stars!",
            "Fantastic! Perfect! Best! Amazing! Incredible value!"
        ]
        
        genuine_reviews = [
            "Good product, works as expected. Delivery was on time.",
            "Decent quality for the price. Setup took some time but works well.",
            "Satisfied with the purchase. Good build quality and reasonable price.",
            "Works well after two weeks of use. Good value for money.",
            "Solid product. Installation was straightforward. Recommended."
        ]
        
        # Prepare training data
        texts = fake_reviews + genuine_reviews
        labels = [1] * len(fake_reviews) + [0] * len(genuine_reviews)  # 1 = fake, 0 = genuine
        
        # Create and train vectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(texts)
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X, labels)
        
        print("Simple fallback model created successfully!")
        return True
    except Exception as e:
        print(f"Error creating fallback model: {e}")
        return False

def load_model_and_vectorizer():
    """Load the trained model and TF-IDF vectorizer"""
    global model, vectorizer
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, 'models')
        
        # Full paths to model files
        model_path = os.path.join(models_dir, 'logistic_regression_model.pkl')
        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
        
        print(f"Looking for models in: {models_dir}")
        print(f"Model path: {model_path}")
        print(f"Vectorizer path: {vectorizer_path}")
        
        # Check if files exist
        if not os.path.exists(model_path):
            print(f"Model file not found at: {model_path}")
            return False
        if not os.path.exists(vectorizer_path):
            print(f"Vectorizer file not found at: {vectorizer_path}")
            return False
        
        # Load the trained Logistic Regression model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load the TF-IDF vectorizer
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        print("Model and vectorizer loaded successfully!")
        return True
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        print("Please ensure the model files exist in the 'models' directory")
        # List all files in current directory for debugging
        try:
            print("Files in current directory:", os.listdir('.'))
            if os.path.exists('models'):
                print("Files in models directory:", os.listdir('models'))
        except:
            pass
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_text(text):
    """Preprocess the input text for prediction"""
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

@app.route('/')
def home():
    """Serve the homepage with the review input form"""
    return render_template('index.html', analytics=analytics_data)

@app.route('/analytics')
def analytics_dashboard():
    """Display analytics dashboard"""
    total = analytics_data['total_predictions']
    fake_percentage = (analytics_data['fake_predictions'] / total * 100) if total > 0 else 0
    genuine_percentage = (analytics_data['genuine_predictions'] / total * 100) if total > 0 else 0
    
    stats = {
        'total_predictions': total,
        'fake_predictions': analytics_data['fake_predictions'],
        'genuine_predictions': analytics_data['genuine_predictions'],
        'predictions_today': analytics_data['predictions_today'],
        'fake_percentage': round(fake_percentage, 1),
        'genuine_percentage': round(genuine_percentage, 1)
    }
    
    return render_template('analytics.html', stats=stats)

@app.route('/predict', methods=['POST'])
def predict():
    """Process the review text and return prediction"""
    try:
        # Get the review text from the form
        review_text = request.form.get('review_text', '').strip()
        
        if not review_text:
            return render_template('index.html', 
                                 error="Please enter a review to analyze.")
        
        # Check if model and vectorizer are loaded
        if model is None or vectorizer is None:
            return render_template('index.html', 
                                 error="Model not loaded. Please contact administrator.")
        
        # Analyze text features
        text_features = analyze_text_features(review_text)
        
        # Preprocess the text
        processed_text = preprocess_text(review_text)
        
        # Transform the text using the TF-IDF vectorizer
        text_vectorized = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        prediction_proba = model.predict_proba(text_vectorized)[0]
        
        # Update analytics
        update_analytics(prediction)
        
        # Get confidence score
        confidence = max(prediction_proba) * 100
        fake_probability = prediction_proba[1] * 100
        genuine_probability = prediction_proba[0] * 100
        
        # Determine result
        if prediction == 1:
            result = "FAKE"
            result_class = "fake"
            description = "This review appears to be fake or suspicious."
            tips = [
                "High confidence in fake classification",
                "Check for overly positive language",
                "Look for generic descriptions",
                "Verify reviewer history if possible"
            ]
        else:
            result = "GENUINE"
            result_class = "genuine"
            description = "This review appears to be genuine."
            tips = [
                "Review shows authentic characteristics",
                "Language appears natural",
                "Details seem specific and realistic",
                "Good balance in sentiment"
            ]
        
        return render_template('index.html', 
                             review_text=review_text,
                             result=result,
                             result_class=result_class,
                             confidence=round(confidence, 2),
                             fake_probability=round(fake_probability, 2),
                             genuine_probability=round(genuine_probability, 2),
                             description=description,
                             text_features=text_features,
                             tips=tips,
                             analytics=analytics_data)
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', 
                             error="An error occurred while processing your request.")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction (JSON response)"""
    try:
        data = request.get_json()
        review_text = data.get('review_text', '').strip()
        
        if not review_text:
            return jsonify({'error': 'No review text provided'}), 400
        
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Preprocess and predict
        processed_text = preprocess_text(review_text)
        text_vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(text_vectorized)[0]
        prediction_proba = model.predict_proba(text_vectorized)[0]
        
        confidence = max(prediction_proba) * 100
        result = "FAKE" if prediction == 1 else "GENUINE"
        
        return jsonify({
            'result': result,
            'confidence': round(confidence, 2),
            'prediction': int(prediction)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load analytics data
    load_analytics()
    
    # Load model and vectorizer on startup
    if load_model_and_vectorizer():
        print("Starting Flask application with trained models...")
    elif create_simple_model():
        print("Starting Flask application with fallback model...")
    else:
        print("Failed to load or create model. Creating basic functionality...")
        # Even if models fail, start the app for basic functionality
    
    # Get port from environment variable (for cloud deployment) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
