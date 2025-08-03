"""
Debug version of the Flask app to troubleshoot model loading
"""

from flask import Flask, render_template, request, jsonify
import pickle
import re
import string
import os
import traceback

app = Flask(__name__)

# Global variables to store the model and vectorizer
model = None
vectorizer = None

def debug_model_loading():
    """Debug the model loading process"""
    global model, vectorizer
    
    print("=== Debug Model Loading ===")
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Check if models directory exists
    models_dir = "models"
    models_path = os.path.join(current_dir, models_dir)
    print(f"Models directory path: {models_path}")
    print(f"Models directory exists: {os.path.exists(models_path)}")
    
    if os.path.exists(models_path):
        print("Contents of models directory:")
        for file in os.listdir(models_path):
            file_path = os.path.join(models_path, file)
            print(f"  - {file} (size: {os.path.getsize(file_path)} bytes)")
    
    # Try to load model
    model_file = os.path.join(models_dir, "logistic_regression_model.pkl")
    vectorizer_file = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    
    print(f"\nTrying to load model from: {model_file}")
    print(f"Model file exists: {os.path.exists(model_file)}")
    
    print(f"Trying to load vectorizer from: {vectorizer_file}")
    print(f"Vectorizer file exists: {os.path.exists(vectorizer_file)}")
    
    try:
        # Load the trained Logistic Regression model
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully!")
        print(f"Model type: {type(model)}")
        
        # Load the TF-IDF vectorizer
        with open(vectorizer_file, 'rb') as f:
            vectorizer = pickle.load(f)
        print("✅ Vectorizer loaded successfully!")
        print(f"Vectorizer type: {type(vectorizer)}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"❌ File not found error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"Traceback: {traceback.format_exc()}")
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
    """Serve the homepage with debug info"""
    debug_info = debug_model_loading()
    
    return f'''
    <h1>🔍 Fake Review Detector - Debug Mode</h1>
    <h2>Debug Information:</h2>
    <p><strong>Current Directory:</strong> {os.getcwd()}</p>
    <p><strong>Models Directory Exists:</strong> {os.path.exists("models")}</p>
    <p><strong>Model Loading Status:</strong> {"✅ Success" if debug_info else "❌ Failed"}</p>
    <p><strong>Model Loaded:</strong> {model is not None}</p>
    <p><strong>Vectorizer Loaded:</strong> {vectorizer is not None}</p>
    
    <h2>Test Form:</h2>
    <form method="POST" action="/predict">
        <textarea name="review_text" rows="4" cols="50" placeholder="Enter review text...">This product is amazing! Perfect! Best ever!</textarea><br><br>
        <button type="submit">Test Prediction</button>
    </form>
    
    <h2>Manual Actions:</h2>
    <a href="/train">🔄 Retrain Model</a> | 
    <a href="/reload">🔃 Reload Models</a> |
    <a href="/check">🔍 Check Files</a>
    '''

@app.route('/check')
def check_files():
    """Check file system"""
    info = []
    info.append(f"Current directory: {os.getcwd()}")
    
    if os.path.exists("models"):
        info.append("Models directory contents:")
        for file in os.listdir("models"):
            file_path = os.path.join("models", file)
            info.append(f"  - {file} ({os.path.getsize(file_path)} bytes)")
    else:
        info.append("❌ Models directory does not exist")
    
    if os.path.exists("sample_reviews.csv"):
        info.append(f"✅ Sample data file exists ({os.path.getsize('sample_reviews.csv')} bytes)")
    else:
        info.append("❌ Sample data file missing")
    
    return "<h1>File Check</h1><pre>" + "\n".join(info) + "</pre><br><a href='/'>← Back</a>"

@app.route('/train')
def train_model():
    """Train the model"""
    try:
        # Import and run training
        exec(open('simple_train.py').read())
        return "<h1>Training Complete</h1><p>Model has been retrained. <a href='/reload'>Reload models</a></p>"
    except Exception as e:
        return f"<h1>Training Failed</h1><p>Error: {e}</p><pre>{traceback.format_exc()}</pre><br><a href='/'>← Back</a>"

@app.route('/reload')
def reload_models():
    """Reload models"""
    success = debug_model_loading()
    if success:
        return "<h1>Models Reloaded</h1><p>✅ Models loaded successfully! <a href='/'>← Back to test</a></p>"
    else:
        return "<h1>Reload Failed</h1><p>❌ Could not load models. <a href='/train'>Try training first</a></p>"

@app.route('/predict', methods=['POST'])
def predict():
    """Process the review text and return prediction"""
    try:
        # Get the review text from the form
        review_text = request.form.get('review_text', '').strip()
        
        if not review_text:
            return "<h1>Error</h1><p>Please enter a review to analyze.</p><a href='/'>← Back</a>"
        
        # Check if model and vectorizer are loaded
        if model is None or vectorizer is None:
            return f"""
            <h1>Model Not Loaded</h1>
            <p>❌ Model: {model is not None}</p>
            <p>❌ Vectorizer: {vectorizer is not None}</p>
            <p><a href='/reload'>Try reloading models</a> | <a href='/train'>Retrain model</a></p>
            <a href='/'>← Back</a>
            """
        
        # Preprocess the text
        processed_text = preprocess_text(review_text)
        print(f"Original text: {review_text}")
        print(f"Processed text: {processed_text}")
        
        # Transform the text using the TF-IDF vectorizer
        text_vectorized = vectorizer.transform([processed_text])
        print(f"Vectorized shape: {text_vectorized.shape}")
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        prediction_proba = model.predict_proba(text_vectorized)[0]
        
        # Get confidence score
        confidence = max(prediction_proba) * 100
        
        # Determine result
        if prediction == 1:
            result = "FAKE"
            description = "This review appears to be fake or suspicious."
        else:
            result = "GENUINE"
            description = "This review appears to be genuine."
        
        return f"""
        <h1>🔍 Prediction Result</h1>
        <p><strong>Review:</strong> {review_text}</p>
        <p><strong>Result:</strong> <span style="color: {'red' if result == 'FAKE' else 'green'}; font-weight: bold;">{result}</span></p>
        <p><strong>Confidence:</strong> {confidence:.2f}%</p>
        <p><strong>Description:</strong> {description}</p>
        <br>
        <a href='/'>← Test Another Review</a>
        """
    
    except Exception as e:
        return f"""
        <h1>Prediction Error</h1>
        <p>Error: {e}</p>
        <pre>{traceback.format_exc()}</pre>
        <a href='/'>← Back</a>
        """

if __name__ == '__main__':
    print("🔍 Starting Fake Review Detector - Debug Mode")
    print("=" * 50)
    
    # Try to load models on startup
    debug_model_loading()
    
    print("\nStarting Flask application on http://localhost:5000")
    print("Open your browser to see debug information")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
