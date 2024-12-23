import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class AudioMoodClassifier:
    def __init__(self):
        self.moods = ['Happy', 'Sad', 'Energetic', 'Calm', 'Angry', 'Peaceful', 'Melancholic', 
                      'Romantic', 'Dark', 'Uplifting', 'Nostalgic', 'Anxious', 'Dreamy', 'Powerful']
        self.scaler = MinMaxScaler()
        
    def classify_mood(self, audio_features):
        # Normalize features
        features_array = np.array([[
            audio_features['energy'],
            audio_features['valence'],
            audio_features['danceability'],
            audio_features['tempo'] / 180.0,  # Normalize tempo
            audio_features['loudness'] / -60.0,  # Normalize loudness
        ]])
        
        # Simple rule-based classification
        energy = features_array[0][0]
        valence = features_array[0][1]
        danceability = features_array[0][2]
        tempo = features_array[0][3]
        loudness = features_array[0][4]
        
        if energy > 0.8 and valence > 0.8:
            return 'Happy'
        elif energy > 0.8 and valence < 0.3:
            return 'Angry'
        elif energy < 0.3 and valence < 0.3:
            return 'Sad'
        elif energy < 0.3 and valence > 0.6:
            return 'Peaceful'
        elif energy > 0.6 and danceability > 0.7:
            return 'Energetic'
        elif tempo < 0.4 and energy < 0.4:
            return 'Calm'
        elif valence < 0.4 and energy < 0.5:
            return 'Melancholic'
        elif valence > 0.6 and tempo < 0.5:
            return 'Romantic'
        elif energy > 0.7 and loudness > 0.7:
            return 'Powerful'
        elif valence > 0.7 and energy > 0.6:
            return 'Uplifting'
        elif energy < 0.4 and valence > 0.4:
            return 'Dreamy'
        else:
            return 'Nostalgic'