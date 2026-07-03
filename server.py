"""
Flask server for the Emotion Detection application
Provides REST API endpoints for emotion analysis and web deployment
"""

from flask import Flask, request, jsonify, render_template
from EmotionDetection.emotion_detection import emotion_detector


app = Flask(__name__)


@app.route('/')
def index():
    """
    Route: / (GET)
    Serves the home page with the emotion detection interface
    """
    return render_template('index.html')


@app.route('/emotionDetector', methods=['POST'])
def emotion_detector_route():
    """
    Route: /emotionDetector (POST)
    
    Flask decorator for the emotion detection API endpoint.
    
    Expected request body (JSON):
    {
        "textToAnalyze": "text to analyze"
    }
    
    Returns formatted response with emotion scores and dominant emotion.
    
    Example:
    Input: {"textToAnalyze": "I love my life"}
    Output: "For the given statement, the system response is 'anger': 0.006274985, 
             'disgust': 0.0025598293, 'fear': 0.009251528, 'joy': 0.9680386 and 
             'sadness': 0.049744144. The dominant emotion is joy."
    """
    
    # Get request data
    request_body = request.get_json()
    
    # Handle missing or invalid request body
    if not request_body:
        return jsonify({
            'error': 'Request body is missing',
            'status_code': 400
        }), 400
    
    # Extract text to analyze from request
    text_to_analyze = request_body.get('textToAnalyze', '')
    
    # Check for blank input - error handling for blank input
    if not text_to_analyze or text_to_analyze.strip() == '':
        return jsonify({
            'error': 'Please enter some text to analyze',
            'status_code': 400
        }), 400
    
    # Perform emotion detection using the emotion_detector function
    emotion_result = emotion_detector(text_to_analyze)
    
    # Check for errors from emotion detector (status_code 400)
    if emotion_result.get('status_code') == 400:
        return jsonify({
            'error': 'Error in emotion detection',
            'status_code': 400
        }), 400
    
    # Format the output as requested by the customer
    # Extract emotion scores
    anger_score = emotion_result.get('anger')
    disgust_score = emotion_result.get('disgust')
    fear_score = emotion_result.get('fear')
    joy_score = emotion_result.get('joy')
    sadness_score = emotion_result.get('sadness')
    dominant_emotion = emotion_result.get('dominant_emotion')
    
    # Create formatted response message
    # Format: "For the given statement, the system response is 'anger': X, 'disgust': Y, 
    #          'fear': Z, 'joy': A and 'sadness': B. The dominant emotion is C."
    formatted_response = (
        f"For the given statement, the system response is 'anger': {anger_score}, "
        f"'disgust': {disgust_score}, 'fear': {fear_score}, 'joy': {joy_score} and "
        f"'sadness': {sadness_score}. The dominant emotion is {dominant_emotion}."
    )
    
    # Return the formatted response
    return jsonify({
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score,
        'dominant_emotion': dominant_emotion,
        'response': formatted_response
    }), 200


@app.errorhandler(400)
def bad_request(error):
    """Handle 400 Bad Request errors"""
    return jsonify({'error': 'Bad request', 'status_code': 400}), 400


@app.errorhandler(404)
def not_found(error):
    """Handle 404 Not Found errors"""
    return jsonify({'error': 'Endpoint not found', 'status_code': 404}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 Internal Server errors"""
    return jsonify({'error': 'Internal server error', 'status_code': 500}), 500


if __name__ == '__main__':
    """
    Deploy the Flask application on localhost:5000
    Run command: python server.py
    Access the application at: http://localhost:5000
    """
    app.run(host='localhost', port=5000, debug=True)
