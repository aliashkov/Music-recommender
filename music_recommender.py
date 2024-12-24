import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from datetime import datetime, timedelta

class EnhancedMusicRecommender:
    def __init__(self, tracks_path='output/tracks.csv', users_path='output/users.csv', listening_path='output/listening_history.csv'):
        self.tracks_df = pd.read_csv(tracks_path)
        self.users_df = pd.read_csv(users_path)
        self.listening_df = pd.read_csv(listening_path)
        self.listening_df['timestamp'] = pd.to_datetime(self.listening_df['timestamp'])

        # Sort listening history by timestamp
        self.listening_df = self.listening_df.sort_values('timestamp', ascending=False)
        
        # Aggregate listening data
        self.aggregated_listening = self._aggregate_listening_data()
        
        self.user_tracks = defaultdict(dict)
        self.track_users = defaultdict(list)
        self.similarity_matrix = None
        self.tracks_matrix = None
        self.track_indices = None
        
        self._initialize_matrices()
        
    def _aggregate_listening_data(self):
        """Aggregate listening data for each user-track combination"""
        # Group by user_id and track_id and aggregate
        aggregated = self.listening_df.groupby(['user_id', 'track_id']).agg({
            'play_count': 'sum',
            'timestamp': 'max',  # Get most recent timestamp
            'user_mood': 'last'  # Get most recent mood
        }).reset_index()
        
        return aggregated
        
    def _initialize_matrices(self):
        """Initialize user-track matrices from aggregated CSV data"""
        for _, row in self.aggregated_listening.iterrows():
            self.add_user_track(row['user_id'], row['track_id'], row['play_count'])
            
    def add_user_track(self, user_id, track_id, play_count):
        """Add or update a track listening record for a user"""
        current_count = self.user_tracks[user_id].get(track_id, 0)
        self.user_tracks[user_id][track_id] = current_count + play_count
        if user_id not in self.track_users[track_id]:
            self.track_users[track_id].append(user_id)

    def get_user_preferences(self, user_id):
        """Calculate user's audio preferences based on recent listening history"""
        # Get recent unique tracks with their latest play count
        recent_listens = self.aggregated_listening[
            self.aggregated_listening['user_id'] == user_id
        ].sort_values('timestamp', ascending=False).head(10)
        
        if recent_listens.empty:
            return None
            
        recent_tracks = self.tracks_df[
            self.tracks_df['track_id'].isin(recent_listens['track_id'])
        ]

        # Weight the preferences by play count
        weighted_preferences = {}
        total_plays = recent_listens['play_count'].sum()

        for feature in ['danceability', 'energy', 'loudness', 'tempo_bpm', 'valence', 'speechiness', 'acousticness']:
            feature_values = []
            for _, track in recent_tracks.iterrows():
                play_count = recent_listens[
                    recent_listens['track_id'] == track['track_id']
                ]['play_count'].iloc[0]
                feature_values.extend([track[feature]] * play_count)
            weighted_preferences[f'avg_{feature}'] = np.mean(feature_values)

        # Get preferred genres weighted by play count
        genre_counts = defaultdict(int)
        for _, track in recent_tracks.iterrows():
            play_count = recent_listens[
                recent_listens['track_id'] == track['track_id']
            ]['play_count'].iloc[0]
            genre_counts[track['genre']] += play_count
            
        preferred_genres = sorted(
            genre_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        weighted_preferences['preferred_genres'] = [genre for genre, _ in preferred_genres]
        
        return weighted_preferences
        
    def get_recent_mood(self, user_id):
        """Get user's most recent mood based on aggregated listening sessions"""
        recent_listens = self.aggregated_listening[
            self.aggregated_listening['user_id'] == user_id
        ].sort_values('timestamp', ascending=False).head(10)
        
        if recent_listens.empty:
            return None
            
        return recent_listens.iloc[0]['user_mood']
    

    def calculate_track_score(self, track, preferences, recent_mood):
        """Calculate recommendation score based on track features and user preferences"""
        # Base audio feature similarity
        feature_score = (
            (1 - abs(track['danceability'] - preferences['avg_danceability'])) * 0.2 +
            (1 - abs(track['valence'] - preferences['avg_valence'])) * 0.2 +
            (1 - abs(track['speechiness'] - preferences['avg_speechiness'])) * 0.2 +
            (1 - abs(track['acousticness'] - preferences['avg_acousticness'])) * 0.2
        )
        
        # Genre matching
        genre_score = 1.0
        if track['genre'] in preferences['preferred_genres']:
            genre_score = 1.3
            
        # Mood matching
        mood_score = 1.0
        if track['mood'] == recent_mood:
            mood_score = 1.2
            
        # Popularity factor
        popularity_score = track['popularity'] / 100.0
        
        # Combined score
        final_score = (
            feature_score * 0.4 +
            genre_score * 0.2 +
            mood_score * 0.2 +
            popularity_score * 0.2
        )
        
        return final_score
        
    def get_recommendations(self, user_id, n_recommendations=5, consider_recent_mood=True):
        """Get personalized recommendations based on recent listening history"""
        # Get user preferences and mood
        preferences = self.get_user_preferences(user_id)
        if not preferences:
            return []
            
        recent_mood = self.get_recent_mood(user_id) if consider_recent_mood else None
        
        # Get recently listened tracks to exclude them
        recent_tracks = set(self.listening_df[
            self.listening_df['user_id'] == user_id
        ].head(10)['track_id'].tolist())
        
        # Calculate scores for all tracks
        recommendations = []
        for _, track in self.tracks_df.iterrows():
            if track['track_id'] not in recent_tracks:
                score = self.calculate_track_score(track, preferences, recent_mood)
                
                recommendations.append({
                    'track_id': track['track_id'],
                    'title': track['title'],
                    'artist': track['artist'],
                    'genre': track['genre'],
                    'mood': track['mood'],
                    'danceability': track['danceability'],
                    'loudness': track['loudness'],
                    'energy': track['energy'],
                    'valence': track['valence'],
                    'acousticness': track['acousticness'],
                    'speechiness': track['speechiness'],
                    'score': score
                })
        
        # Sort by score and return top N recommendations
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:n_recommendations]