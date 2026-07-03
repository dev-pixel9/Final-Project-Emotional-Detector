"""
Flask server for the Emotion Detection application.
Provides REST API endpoints for emotion analysis and web deployment.
"""

from flask import Flask, request, jsonify, render_template
from EmotionDetection.emotion_detection import emotion_detector

app = Flask(__name__)


@app.route('/')
def render_index_page():
    """
    Route: / (GET)
    Serves the home page with the emotion detection interface.
    """
    return render_template('index.html')


@app.route('/emotionDetector', methods=['GET', 'POST'])
@app.route('/emotion_detector', methods=['GET', 'POST'])
def sent_detector():
    """
    Route handler to detect emotions from input text.
    Handles GET query parameters and POST JSON requests.
    """
    # 1. Retrieve the text to analyze
    text_to_analyze = ""
    is_json_request = False

    if request.method == 'POST':
        if request.is_json:
            request_body = request.get_json()
            if request_body:
                text_to_analyze = request_body.get('textToAnalyze', '')
                is_json_request = True
        else:
            text_to_analyze = request.form.get('textToAnalyze', '')
    else:
        text_to_analyze = request.args.get('textToAnalyze', '')

    # 2. Check for blank/empty input
    if not text_to_analyze or text_to_analyze.strip() == '':
        error_msg = "Invalid text! Please try again!"
        if is_json_request:
            return jsonify({
                'error': error_msg,
                'status_code': 400
            }), 400
        return error_msg, 400

    # 3. Call the emotion detector
    emotion_result = emotion_detector(text_to_analyze)

    # 4. Check for errors
    if emotion_result.get('status_code') == 400 or emotion_result.get('dominant_emotion') is None:
        error_msg = "Invalid text! Please try again!"
        if is_json_request:
            return jsonify({
                'error': error_msg,
                'status_code': 400
            }), 400
        return error_msg, 400

    # 5. Extract scores
    anger_score = emotion_result.get('anger')
    disgust_score = emotion_result.get('disgust')
    fear_score = emotion_result.get('fear')
    joy_score = emotion_result.get('joy')
    sadness_score = emotion_result.get('sadness')
    dominant_emotion = emotion_result.get('dominant_emotion')

    # 6. Format the string output
    formatted_response = (
        f"For the given statement, the system response is 'anger': {anger_score}, "
        f"'disgust': {disgust_score}, 'fear': {fear_score}, 'joy': {joy_score} and "
        f"'sadness': {sadness_score}. The dominant emotion is {dominant_emotion}."
    )

    if is_json_request:
        return jsonify({
            'anger': anger_score,
            'disgust': disgust_score,
            'fear': fear_score,
            'joy': joy_score,
            'sadness': sadness_score,
            'dominant_emotion': dominant_emotion,
            'response': formatted_response
        }), 200

    return formatted_response, 200


@app.errorhandler(400)
def bad_request(_error):
    """Handle 400 Bad Request errors."""
    return jsonify({'error': 'Bad request', 'status_code': 400}), 400


@app.errorhandler(404)
def not_found(_error):
    """Handle 404 Not Found errors."""
    return jsonify({'error': 'Endpoint not found', 'status_code': 404}), 404


@app.errorhandler(500)
def internal_error(_error):
    """Handle 500 Internal Server errors."""
    return jsonify({'error': 'Internal server error', 'status_code': 500}), 500


if __name__ == '__main__':
    # Deploy the Flask application on localhost:5000
    # Run command: python server.py
    # Access the application at: http://localhost:5000
    app.run(host='localhost', port=5000, debug=True)
