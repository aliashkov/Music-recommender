import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from datetime import datetime, timedelta
import librosa

class EnhancedMusicRecommender:
    def __init__(self, tracks_path='output/tracks.csv', users_path='output/users.csv', listening_path='output/listening_history.csv'):
        self.tracks_df = pd.read_csv(tracks_path)
        self.users_df = pd.read_csv(users_path)
        self.listening_df = pd.read_csv(listening_path)
        self.listening_df['timestamp'] = pd.to_datetime(self.listening_df['timestamp'])
        
        # Normalize energy and brightness to 0-1 range
        self._normalize_audio_features()
        
        self.user_tracks = defaultdict(list)
        self.track_users = defaultdict(list)
        self.similarity_matrix = None
        self.tracks_matrix = None
        self.track_indices = None
        
        self._initialize_matrices()

    def _normalize_audio_features(self):
        """Normalize audio features to 0-1 range"""
        features_to_normalize = ['energy', 'brightness']
        for feature in features_to_normalize:
            if feature in self.tracks_df.columns:
                min_val = self.tracks_df[feature].min()
                max_val = self.tracks_df[feature].max()
                self.tracks_df[feature] = (self.tracks_df[feature] - min_val) / (max_val - min_val)

    def _initialize_matrices(self):
        """Initialize user-track matrices with recent listening history"""
        # Sort listening history by timestamp
        sorted_listening = self.listening_df.sort_values('timestamp', ascending=False)
        
        # Group by user and take last 100 listens
        for user_id in self.users_df['user_id']:
            user_history = sorted_listening[sorted_listening['user_id'] == user_id].head(100)
            for _, row in user_history.iterrows():
                self.add_user_track(row['user_id'], row['track_id'], row)

    def determine_track_mood(self, track_features):
        """Determine track mood based on audio features"""
        # Enhanced mood determination logic
        valence = track_features.get('valence', 0)
        energy = track_features.get('energy', 0)
        danceability = track_features.get('danceability', 0)
        
        if valence > 0.7 and energy > 0.7:
            return 'Happy'
        elif valence < 0.3 and energy < 0.4:
            return 'Sad'
        elif energy > 0.8:
            return 'Energetic'
        elif valence < 0.4 and energy > 0.6:
            return 'Angry'
        elif valence > 0.6 and energy < 0.4:
            return 'Peaceful'
        elif danceability > 0.7:
            return 'Uplifting'
        elif valence < 0.5 and energy < 0.5:
            return 'Melancholic'
        else:
            return 'Neutral'

    def add_user_track(self, user_id, track_id, listening_data):
        """Add a track listening record with timestamp and context"""
        self.user_tracks[user_id].append({
            'track_id': track_id,
            'timestamp': listening_data['timestamp'],
            'time_of_day': listening_data['time_of_day'],
            'user_mood': listening_data['user_mood']
        })
        self.track_users[track_id].append(user_id)

    def get_user_current_context(self, user_id):
        """Get user's current context based on recent listening patterns"""
        if not self.user_tracks[user_id]:
            return None

        recent_listens = sorted(self.user_tracks[user_id], 
                              key=lambda x: x['timestamp'], 
                              reverse=True)[:10]

        # Analyze time patterns
        current_hour = datetime.now().hour
        time_preference = defaultdict(int)
        mood_preference = defaultdict(int)

        for listen in recent_listens:
            listen_hour = int(listen['time_of_day'].split(':')[0])
            time_preference[listen_hour] += 1
            mood_preference[listen['user_mood']] += 1

        return {
            'preferred_time': max(time_preference.items(), key=lambda x: x[1])[0],
            'current_mood': max(mood_preference.items(), key=lambda x: x[1])[0],
            'time_of_day': current_hour
        }
    
    def get_recent_mood(self, user_id):
      """Get user's most recent mood based on listening history"""
      if user_id not in self.user_tracks:
        return "Unknown"
    
      recent_listens = sorted(self.user_tracks[user_id], 
                          key=lambda x: x['timestamp'], 
                          reverse=True)
    
      if not recent_listens:
        return "Unknown"
    
      return recent_listens[0]['user_mood']

    def calculate_context_similarity(self, track_data, user_context):
        """Calculate similarity score based on contextual factors"""
        if not user_context:
            return 1.0

        context_score = 1.0
        
        # Time of day compatibility
        hour_diff = abs(user_context['preferred_time'] - user_context['time_of_day'])
        time_factor = 1 - (hour_diff / 24)
        context_score *= (0.8 + 0.2 * time_factor)

        # Mood compatibility
        track_mood = self.determine_track_mood(track_data)
        mood_match = 1.2 if track_mood == user_context['current_mood'] else 0.8
        context_score *= mood_match

        return context_score

    def get_recommendations(self, user_id, n_recommendations=5):
        """Get personalized track recommendations"""
        if user_id not in self.user_tracks:
            return []

        user_context = self.get_user_current_context(user_id)
        
        # Get user's recently played tracks
        recent_tracks = set(listen['track_id'] for listen in self.user_tracks[user_id])
        
        # Calculate recommendation scores
        recommendations = []
        for _, track in self.tracks_df.iterrows():
            if track['track_id'] not in recent_tracks:
                # Base score
                score = track['popularity'] / 100.0
                
                # Audio features score
                audio_score = (
                    track['danceability'] * 0.3 +
                    track['energy'] * 0.2 +
                    track['valence'] * 0.2 +
                    (1 - track['speechiness']) * 0.15 +
                    track['acousticness'] * 0.15
                )
                
                # Context score
                context_score = self.calculate_context_similarity(track, user_context)
                
                # Final score
                final_score = (score * 0.3 + audio_score * 0.4 + context_score * 0.3)
                
                recommendations.append({
                    'track_id': track['track_id'],
                    'title': track['title'],
                    'artist': track['artist'],
                    'genre': track['genre'],
                    'mood': self.determine_track_mood(track),
                    'danceability': track['danceability'],
                    'energy': track['energy'],
                    'valence': track['valence'],
                    'duration': track['duration_sec'],
                    'score': final_score
                })

        # Sort and return top recommendations
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:n_recommendations]

    def add_track_duration(self, audio_path):
        """Calculate accurate track duration"""
        try:
            y, sr = librosa.load(audio_path)
            duration = librosa.get_duration(y=y, sr=sr)
            return int(duration)
        except Exception as e:
            print(f"Error calculating duration for {audio_path}: {e}")
            return None