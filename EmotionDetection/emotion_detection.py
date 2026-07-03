"""
Emotion Detection module using Watson NLP Library
This module analyzes text and detects emotions using IBM Watson NLP.
"""

import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_cloud_sdk_core.get_authenticator import get_authenticator_from_environment


def emotion_detector(text_to_analyze):
    """
    Detects emotions in the provided text using Watson NLP.
    
    This function takes input text and returns emotion scores for five emotions:
    anger, disgust, fear, joy, and sadness. It also identifies the dominant emotion
    (the one with the highest score).
    
    Error Handling: For status_code = 400 (blank/invalid input), the function returns
    the same dictionary, but with values for all keys being None.
    
    Args:
        text_to_analyze (str): The text to analyze for emotions
        
    Returns:
        dict: A dictionary with the following format:
        {
            'anger': float,
            'disgust': float,
            'fear': float,
            'joy': float,
            'sadness': float,
            'dominant_emotion': str
        }
        
        Returns error format with status_code 400 if input is invalid:
        {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None,
            'status_code': 400
        }
    """
    
    # Error Handling: Validate input - check for None, empty string, or whitespace-only input
    # For status_code = 400 (blank entries), return dictionary with all values as None
    if not text_to_analyze or text_to_analyze.strip() == "":
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None,
            'status_code': 400
        }
    
    # Set up Watson NLU authenticator using environment credentials
    authenticator = get_authenticator_from_environment('WATSON_NLU')
    nlu = NaturalLanguageUnderstandingV1(
        version='2021-08-01',
        authenticator=authenticator,
        service_url='https://api.us-south.natural-language-understanding.watson.cloud.ibm.com'
    )
    
    # Perform emotion analysis using Watson NLP
    try:
        # Call Watson NLP analyze method with emotion feature
        response = nlu.analyze(
            text=text_to_analyze,
            features={
                'emotion': {}
            }
        ).get_result()
        
        # Convert response text into a dictionary using json library functions
        # The response is already in dictionary format from Watson NLP
        # Extract the emotion scores from the response
        emotion_scores = response['emotion']['document']['emotion']
        
        # Extract the required set of emotions with their scores
        # The emotion_scores dictionary contains: anger, disgust, fear, joy, sadness
        anger_score = emotion_scores.get('anger')
        disgust_score = emotion_scores.get('disgust')
        fear_score = emotion_scores.get('fear')
        joy_score = emotion_scores.get('joy')
        sadness_score = emotion_scores.get('sadness')
        
        # Write code logic to find the dominant emotion
        # The dominant emotion is the one with the highest score
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        
        # Return the formatted output with all emotions and dominant emotion
        return {
            'anger': anger_score,
            'disgust': disgust_score,
            'fear': fear_score,
            'joy': joy_score,
            'sadness': sadness_score,
            'dominant_emotion': dominant_emotion
        }
    
    except Exception as exception:
        # Error Handling: Return error response with status_code 400 for any API failures
        # All emotion values are set to None
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None,
            'status_code': 400
        }
