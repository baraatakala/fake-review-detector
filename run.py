#!/usr/bin/env python3
"""
Quick setup and run script for Fake Review Detector
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def train_model():
    """Train the machine learning model"""
    print("\nTraining the machine learning model...")
    try:
        subprocess.check_call([sys.executable, "train_model.py"])
        print("✅ Model trained successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error training model: {e}")
        return False

def run_app():
    """Run the Flask application"""
    print("\nStarting Flask application...")
    try:
        subprocess.check_call([sys.executable, "app.py"])
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running application: {e}")
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")

def main():
    """Main setup and run function"""
    print("🔍 Fake Review Detector Setup")
    print("=" * 40)
    
    # Check if models directory exists
    if not os.path.exists("models"):
        print("📦 Models directory not found. Setting up for the first time...")
        
        # Install requirements
        if not install_requirements():
            return
        
        # Train model
        if not train_model():
            return
    else:
        print("✅ Models found. Skipping training step.")
    
    # Run the application
    run_app()

if __name__ == "__main__":
    main()
