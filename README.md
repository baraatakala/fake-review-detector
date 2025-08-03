# Fake Review Detector

A Flask web application that uses machine learning to detect fake product reviews using TF-IDF vectorization and Logistic Regression.

## Features

- **Web Interface**: Simple form to input review text and get predictions
- **Machine Learning**: Uses TF-IDF vectorization and Logistic Regression
- **Text Preprocessing**: Comprehensive text cleaning and preprocessing
- **API Endpoint**: JSON API for programmatic access
- **Real-time Predictions**: Instant classification with confidence scores

## Project Structure

```
fake_review_detector/
├── app.py                 # Flask web application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── models/               # Directory for saved models (created after training)
│   ├── logistic_regression_model.pkl
│   └── tfidf_vectorizer.pkl
├── templates/            # HTML templates
│   └── index.html        # Main page template
└── static/              # Static files (CSS, JS)
    └── style.css        # Stylesheet
```

## Installation

1. **Clone or download this project**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Train the Model

Before running the web application, you need to train the machine learning model:

```bash
python train_model.py
```

This script will:
- Create a sample dataset (or use your own CSV file)
- Preprocess the text data
- Train a Logistic Regression model with TF-IDF features
- Save the trained model and vectorizer to the `models/` directory
- Display model performance metrics

### Step 2: Run the Web Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Step 3: Use the Application

1. Open your web browser and go to `http://localhost:5000`
2. Enter a product review in the text area
3. Click "Analyze Review"
4. View the prediction result (Fake/Genuine) with confidence score

## API Usage

The application also provides a JSON API endpoint:

```bash
POST /api/predict
Content-Type: application/json

{
    "review_text": "This product is amazing! Best ever! Perfect! 5 stars!"
}
```

Response:
```json
{
    "result": "FAKE",
    "confidence": 85.67,
    "prediction": 1
}
```

## Using Your Own Dataset

To use your own dataset instead of the sample data:

1. Create a CSV file with columns: `review_text` and `label`
   - `review_text`: The review text
   - `label`: 0 for genuine reviews, 1 for fake reviews

2. Save the file as `sample_reviews.csv` in the project directory, or modify the filename in `train_model.py`

3. Run the training script:
   ```bash
   python train_model.py
   ```

## Model Details

- **Algorithm**: Logistic Regression with balanced class weights
- **Vectorization**: TF-IDF with 5000 max features, unigrams and bigrams
- **Preprocessing**: 
  - Text lowercasing
  - URL and email removal
  - Special character removal
  - Stop word removal (when NLTK is available)
  - Extra whitespace normalization

## Performance

The model performance depends on the quality and size of your training data. With the sample dataset, you can expect:
- Basic functionality demonstration
- Simple pattern recognition for obviously fake vs. genuine reviews

For production use, train with a larger, more diverse dataset.

## File Descriptions

### `app.py`
The main Flask application with:
- Homepage route serving the input form
- Prediction route for processing reviews
- API endpoint for JSON responses
- Error handling and model loading

### `train_model.py`
The model training script that:
- Creates or loads training data
- Preprocesses text data
- Trains a Logistic Regression model
- Evaluates model performance
- Saves the trained model and vectorizer

### `templates/index.html`
The web interface featuring:
- Clean, responsive design
- Review input form
- Result display with styling
- Error message handling

## Customization

### Modify Text Preprocessing
Edit the `preprocess_text()` function in both `app.py` and `train_model.py` to customize text cleaning.

### Change Model Parameters
Modify the `TfidfVectorizer` and `LogisticRegression` parameters in `train_model.py`:

```python
# TF-IDF parameters
self.vectorizer = TfidfVectorizer(
    max_features=10000,     # Increase for more features
    min_df=3,               # Minimum document frequency
    max_df=0.7,             # Maximum document frequency
    ngram_range=(1, 3),     # Include trigrams
    stop_words='english'
)

# Model parameters
self.model = LogisticRegression(
    random_state=42,
    max_iter=2000,          # More iterations
    C=0.5,                  # Regularization strength
    class_weight='balanced'
)
```

### Styling
Modify `static/style.css` to customize the web interface appearance.

## Troubleshooting

1. **Model files not found**: Run `python train_model.py` first
2. **NLTK data missing**: The script will download required data automatically
3. **Memory issues**: Reduce `max_features` in TF-IDF vectorizer
4. **Port conflicts**: Change the port in `app.py`: `app.run(port=5001)`

## Dependencies

- **Flask**: Web framework
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **nltk**: Natural language processing
- **pickle**: Model serialization

## License

This project is for educational and demonstration purposes.
