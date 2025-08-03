import subprocess
import sys
import os

# Change to the correct directory
os.chdir(r"c:\Users\isc\VS_Code\fake_review_detector")

# Start the Flask app
print("Starting Flask application...")
print("Open your browser to: http://localhost:5000")
print("Press Ctrl+C to stop the server")

try:
    subprocess.run([sys.executable, "app.py"])
except KeyboardInterrupt:
    print("\nFlask application stopped.")
