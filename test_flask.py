from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return '''
    <h1>🔍 Fake Review Detector - Test Server</h1>
    <p>This is a test server to verify Flask is working.</p>
    <p>If you can see this, Flask is running correctly!</p>
    <a href="http://localhost:5000">Click here once the main app is running</a>
    '''

if __name__ == '__main__':
    print("Starting test Flask server...")
    print("Open http://localhost:5001 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5001)
