"""
Simple test to verify models work and start Flask app
"""

import pickle
import os

print("🔍 Testing Fake Review Detector Models")
print("=" * 50)

# Test 1: Check files exist
print("1. Checking model files...")
model_exists = os.path.exists('models/logistic_regression_model.pkl')
vectorizer_exists = os.path.exists('models/tfidf_vectorizer.pkl')

print(f"   Model file: {'✅' if model_exists else '❌'}")
print(f"   Vectorizer file: {'✅' if vectorizer_exists else '❌'}")

if not (model_exists and vectorizer_exists):
    print("❌ Model files missing. Running quick fix...")
    exec(open('quick_fix.py').read())

# Test 2: Load models
print("\n2. Loading models...")
try:
    with open('models/logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("   ✅ Model loaded successfully")
    
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("   ✅ Vectorizer loaded successfully")
    
except Exception as e:
    print(f"   ❌ Error loading models: {e}")
    print("   Running quick fix...")
    exec(open('quick_fix.py').read())

# Test 3: Test prediction
print("\n3. Testing prediction...")
try:
    test_text = "This product is amazing! Perfect! Best ever!"
    processed_text = test_text.lower()
    text_vectorized = vectorizer.transform([processed_text])
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0]
    
    result = "FAKE" if prediction == 1 else "GENUINE"
    confidence = max(probability) * 100
    
    print(f"   ✅ Test successful!")
    print(f"   Text: '{test_text}'")
    print(f"   Result: {result}")
    print(f"   Confidence: {confidence:.1f}%")
    
except Exception as e:
    print(f"   ❌ Prediction failed: {e}")

# Test 4: Import Flask app
print("\n4. Testing Flask app import...")
try:
    from app import app, load_model_and_vectorizer
    print("   ✅ Flask app imported successfully")
    
    if load_model_and_vectorizer():
        print("   ✅ App models loaded successfully")
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYou can now run: python app.py")
        print("Then open: http://localhost:5000")
    else:
        print("   ❌ App model loading failed")
        
except Exception as e:
    print(f"   ❌ Flask app error: {e}")

print("\n" + "=" * 50)
