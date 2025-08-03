"""
Test script to verify the fake review detector is working
"""

import os
import sys

def test_imports():
    """Test if all required packages are available"""
    print("Testing imports...")
    try:
        import flask
        print("✅ Flask imported successfully")
        
        import sklearn
        print("✅ Scikit-learn imported successfully")
        
        import pandas
        print("✅ Pandas imported successfully")
        
        import numpy
        print("✅ NumPy imported successfully")
        
        import pickle
        print("✅ Pickle imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_files():
    """Check if model files exist"""
    print("\nChecking model files...")
    
    model_dir = "models"
    model_file = os.path.join(model_dir, "logistic_regression_model.pkl")
    vectorizer_file = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    
    if os.path.exists(model_file):
        print("✅ Model file found")
    else:
        print("❌ Model file not found")
        return False
    
    if os.path.exists(vectorizer_file):
        print("✅ Vectorizer file found")
    else:
        print("❌ Vectorizer file not found")
        return False
    
    return True

def test_model_loading():
    """Test loading and using the model"""
    print("\nTesting model loading...")
    
    try:
        import pickle
        
        # Load model
        with open('models/logistic_regression_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully")
        
        # Load vectorizer
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("✅ Vectorizer loaded successfully")
        
        # Test prediction
        test_text = "This product is amazing! Perfect! Best ever!"
        processed_text = test_text.lower()
        text_vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0]
        
        result = "FAKE" if prediction == 1 else "GENUINE"
        confidence = max(probability) * 100
        
        print(f"✅ Test prediction successful:")
        print(f"   Text: '{test_text}'")
        print(f"   Result: {result}")
        print(f"   Confidence: {confidence:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def test_flask_app():
    """Test if Flask app can be imported and configured"""
    print("\nTesting Flask app...")
    
    try:
        # Try to import the app
        from app import app, load_model_and_vectorizer
        print("✅ Flask app imported successfully")
        
        # Test model loading function
        if load_model_and_vectorizer():
            print("✅ App model loading successful")
        else:
            print("❌ App model loading failed")
            return False
        
        print("✅ Flask app is ready to run!")
        print("   Run 'python app.py' to start the web server")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Flask app: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 Fake Review Detector - System Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test model files
    if not test_model_files():
        all_tests_passed = False
    
    # Test model loading
    if not test_model_loading():
        all_tests_passed = False
    
    # Test Flask app
    if not test_flask_app():
        all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! The application is ready to use.")
        print("\nTo start the web application:")
        print("1. Run: python app.py")
        print("2. Open your browser to: http://localhost:5000")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return all_tests_passed

if __name__ == "__main__":
    main()
