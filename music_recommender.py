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
        
        self.user_tracks = defaultdict(dict)
        self.track_users = defaultdict(list)
        self.similarity_matrix = None
        self.tracks_matrix = None
        self.track_indices = None
        
        self._initialize_matrices()
        
    def _initialize_matrices(self):
        """Initialize user-track matrices from CSV data"""
        for _, row in self.listening_df.iterrows():
            self.add_user_track(row['user_id'], row['track_id'], row['play_count'])
            
    def add_user_track(self, user_id, track_id, play_count=1):
        """Add a track listening record for a user"""
        self.user_tracks[user_id][track_id] = play_count
        self.track_users[track_id].append(user_id)
        
    def build_matrix(self):
        """Build user-track matrix with enhanced features"""
        users = list(self.user_tracks.keys())
        tracks = list(set([track for user in self.user_tracks.values() for track in user.keys()]))
        
        self.track_indices = {track: idx for idx, track in enumerate(tracks)}
        self.tracks_matrix = np.zeros((len(users), len(tracks)))
        
        for i, user in enumerate(users):
            user_data = self.users_df[self.users_df['user_id'] == user].iloc[0]
            
            for track, count in self.user_tracks[user].items():
                j = self.track_indices[track]
                track_data = self.tracks_df[self.tracks_df['track_id'] == track].iloc[0]
                
                # Calculate weighted score
                base_score = count
                
                # Genre preference bonus
                #if track_data['genre'] == user_data['preferred_genre']:
                #    base_score *= 1.2
                    
                # Audio feature weights
                audio_score = (
                    track_data['danceability'] * 0.2 +
                    track_data['energy'] * 0.2 +
                    (track_data['loudness'] / -60) * 0.1  # Normalize loudness
                )
                
                # Popularity factor
                popularity_weight = track_data['popularity'] / 100
                
                # Combined score
                final_score = base_score * (0.6 + 0.2 * popularity_weight + 0.2 * audio_score)
                
                self.tracks_matrix[i][j] = final_score
                
        self.similarity_matrix = cosine_similarity(self.tracks_matrix.T)

    def get_user_preferences(self, user_id):
        """Calculate user's audio preferences based on recent listening history"""
        recent_listens = self.listening_df[
            self.listening_df['user_id'] == user_id
        ].head(10)

        print(recent_listens)
        
        if recent_listens.empty:
            return None
            
        recent_tracks = self.tracks_df[
            self.tracks_df['track_id'].isin(recent_listens['track_id'])
        ]

        print(recent_tracks)
        
        preferences = {
            'avg_danceability': recent_tracks['danceability'].mean(),
            'avg_energy': recent_tracks['energy'].mean(),
            'avg_loudness': recent_tracks['loudness'].mean(),
            'preferred_genres': recent_tracks['genre'].value_counts().head(3).index.tolist(),
            'avg_tempo': recent_tracks['tempo_bpm'].mean(),
            'avg_valence': recent_tracks['valence'].mean(),
            'avg_speechiness': recent_tracks['speechiness'].mean(),
            'avg_acousticness': recent_tracks['acousticness'].mean(),
        }
        
        return preferences
        
    def get_recent_mood(self, user_id):
        """Get user's most recent mood based on last 10 listening sessions"""
        recent_listens = self.listening_df[
            self.listening_df['user_id'] == user_id
        ].head(10)

        print(recent_listens)
        
        if recent_listens.empty:
            return None
            
        return recent_listens.iloc[0]['user_mood']
    

    def calculate_track_score(self, track, preferences, recent_mood):
        """Calculate recommendation score based on track features and user preferences"""
        # Base audio feature similarity
        feature_score = (
            (1 - abs(track['danceability'] - preferences['avg_danceability'])) * 0.15 +
            (1 - abs(track['energy'] - preferences['avg_energy'])) * 0.15 +
            (1 - abs(track['valence'] - preferences['avg_valence'])) * 0.1 +
            (1 - abs(track['speechiness'] - preferences['avg_speechiness'])) * 0.1 +
            (1 - abs(track['acousticness'] - preferences['avg_acousticness'])) * 0.1
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