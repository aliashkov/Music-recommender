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
        
    def get_recent_mood(self, user_id, hours=24):
        """Analyze user's recent listening mood"""
        recent_cutoff = datetime.now()
        
        recent_listening = self.listening_df[
            (self.listening_df['user_id'] == user_id) &
            (self.listening_df['timestamp'] >= recent_cutoff - timedelta(hours=hours))
        ]
        
        if len(recent_listening) == 0:
            return None
            
        mood_counts = recent_listening['user_mood'].value_counts()
        return mood_counts.index[0] if len(mood_counts) > 0 else None
        
    def get_recommendations(self, user_id, n_recommendations=5, consider_recent_mood=True):
        """Get track recommendations considering recent mood and enhanced features"""
        if user_id not in self.user_tracks:
            return []
            
        if self.similarity_matrix is None:
            self.build_matrix()
            
        user_tracks = self.user_tracks[user_id]
        user_track_indices = [self.track_indices[track] for track in user_tracks]
        
        # Calculate base recommendation scores
        scores = np.zeros(len(self.track_indices))
        for track_idx in user_track_indices:
            scores += self.similarity_matrix[track_idx]
            
        # Apply mood-based filtering
        if consider_recent_mood:
            recent_mood = self.get_recent_mood(user_id)
            if recent_mood:
                track_ids = list(self.track_indices.keys())
                mood_multiplier = np.ones(len(scores))
                
                for i, track_id in enumerate(track_ids):
                    track_data = self.tracks_df[self.tracks_df['track_id'] == track_id]
                    if not track_data.empty:
                        track_mood = track_data.iloc[0]['mood']
                        if track_mood == recent_mood:
                            mood_multiplier[i] = 1.3
                
                scores *= mood_multiplier
        
        # Prepare recommendations
        track_ids = list(self.track_indices.keys())
        recommendations = []
        sorted_indices = np.argsort(scores)[::-1]
        
        for idx in sorted_indices:
            track_id = track_ids[idx]
            if track_id not in user_tracks:
                track_info = self.tracks_df[self.tracks_df['track_id'] == track_id].iloc[0]
                recommendations.append({
                    'track_id': track_id,
                    'title': track_info['title'],
                    'artist': track_info['artist'],
                    'genre': track_info['genre'],
                    'mood': track_info['mood'],
                    'danceability': track_info['danceability'],
                    'loudness': track_info['loudness'],
                    'energy': track_info['energy'],
                    'score': scores[idx]
                })
                if len(recommendations) >= n_recommendations:
                    break
                    
        return recommendations