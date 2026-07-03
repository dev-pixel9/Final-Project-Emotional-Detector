"""
Unit tests for the Emotion Detection application
This test module validates that the emotion_detector function correctly
identifies the dominant emotion for various test statements.
"""

import unittest
from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetection(unittest.TestCase):
    """Test cases for the emotion_detector function"""
    
    def test_emotion_detector_joy(self):
        """
        Test Case 1: Verify that 'I am glad this happened' returns dominant_emotion as 'joy'
        """
        result = emotion_detector("I am glad this happened")
        self.assertEqual(result['dominant_emotion'], 'joy')
    
    def test_emotion_detector_anger(self):
        """
        Test Case 2: Verify that 'I am really mad about this' returns dominant_emotion as 'anger'
        """
        result = emotion_detector("I am really mad about this")
        self.assertEqual(result['dominant_emotion'], 'anger')
    
    def test_emotion_detector_disgust(self):
        """
        Test Case 3: Verify that 'I feel disgusted just hearing about this' 
        returns dominant_emotion as 'disgust'
        """
        result = emotion_detector("I feel disgusted just hearing about this")
        self.assertEqual(result['dominant_emotion'], 'disgust')
    
    def test_emotion_detector_sadness(self):
        """
        Test Case 4: Verify that 'I am so sad about this' returns dominant_emotion as 'sadness'
        """
        result = emotion_detector("I am so sad about this")
        self.assertEqual(result['dominant_emotion'], 'sadness')
    
    def test_emotion_detector_fear(self):
        """
        Test Case 5: Verify that 'I am really afraid that this will happen' 
        returns dominant_emotion as 'fear'
        """
        result = emotion_detector("I am really afraid that this will happen")
        self.assertEqual(result['dominant_emotion'], 'fear')


if __name__ == '__main__':
    unittest.main()
