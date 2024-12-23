import pandas as pd
import numpy as np
import datetime
import librosa
import librosa.feature
import os
from pathlib import Path

def extract_audio_features(audio_path):
    """Extract audio features from MP3 file"""
    try:
        print(f"Processing: {audio_path}")
        y, sr = librosa.load(audio_path)
        
        # Basic Features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
        
        # Rhythm features
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        
        # Energy/loudness
        rms = librosa.feature.rms(y=y).mean()
        
        # MFCC features (more coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_means = mfccs.mean(axis=1)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_means = chroma.mean(axis=1)
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_means = contrast.mean(axis=1)
        
        # Harmony features
        harmonic = librosa.effects.harmonic(y)
        harmonic_mean = np.mean(harmonic)
        
        # Calculate more complex features
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
        
        # Danceability
        danceability = float(np.mean(pulse) * np.mean(rms) * 10)

        print("danceability:", danceability)
        
        # Valence (emotional content) - simplified approximation
        valence = float((harmonic_mean + np.mean(chroma_means)) / 2)
        valence = min(max(valence, 0), 1)

        print("valence:", valence)
        
        # Speechiness approximation
        speechiness = float(np.mean(mfcc_means[1:5]) / 100)  # Using early MFCCs
        speechiness = min(max(speechiness, 0), 1)

        print("speechiness:", speechiness)
        
        # Acousticness approximation
        acousticness = 1 - (float(spectral_rolloff) / (sr/2))

        print("acousticness:", acousticness)
        
        # Instrumentalness approximation
        instrumentalness = float(np.mean(contrast_means) / 100)


        print("Instrumentals:", instrumentalness)
        
        print(f"Features extracted successfully for {audio_path}")
        
        return {
            'danceability': danceability,
            'loudness': float(rms),
            'tempo': float(tempo),
            'energy': float(spectral_rolloff),
            'brightness': float(spectral_centroids),
            'valence': valence,  # emotional positiveness
            'speechiness': speechiness,
            'acousticness': acousticness,
            'instrumentalness': instrumentalness,
            'zero_crossing_rate': float(zero_crossing_rate),
            'spectral_bandwidth': float(spectral_bandwidth),
            'harmonic_mean': float(harmonic_mean),
        }
    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        return None

def create_dataset(audio_dir="audio_files"):
    np.random.seed(42)
    n_tracks = 40  # Increased number of tracks
    n_users = 20   # Increased number of users
    n_listening_records = 250  # Increased number of records

    # Extended lists of categories
    genres = ['Rock', 'Pop', 'Hip Hop', 'Jazz', 'Classical', 'Electronic', 'R&B', 'Metal', 
              'Folk', 'Latin', 'Blues', 'Country', 'World', 'Reggae', 'Funk', 'Soul']
    
    moods = ['Happy', 'Sad', 'Energetic', 'Calm', 'Angry', 'Peaceful', 'Melancholic', 
             'Romantic', 'Dark', 'Uplifting', 'Nostalgic', 'Anxious', 'Dreamy', 'Powerful']
    
    decades = ['1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s']
    
    occasions = ['Party', 'Workout', 'Study', 'Sleep', 'Meditation', 'Road Trip', 
                'Work', 'Dating', 'Relaxation', 'Gaming']

    # Initialize tracks data with placeholder values
    tracks_data = {
        'track_id': [f'track_{i}' for i in range(1, n_tracks + 1)],
        'title': [f'Song Title {i}' for i in range(1, n_tracks + 1)],
        'artist': [f'Artist {np.random.randint(1, 51)}' for _ in range(n_tracks)],
        'genre': np.random.choice(genres, n_tracks),
        'subgenre': [f'{g} - {np.random.choice(["Type1", "Type2", "Type3"])}' for g in np.random.choice(genres, n_tracks)],
        'mood': np.random.choice(moods, n_tracks),
        'occasion': np.random.choice(occasions, n_tracks),
        'decade': np.random.choice(decades, n_tracks),
        'tempo_bpm': np.random.randint(60, 180, n_tracks),
        'popularity': np.random.randint(1, 101, n_tracks),
        'duration_sec': np.random.randint(120, 500, n_tracks),
        'danceability': np.random.uniform(0, 1, n_tracks),
        'loudness': np.random.uniform(-60, 0, n_tracks),
        'energy': np.random.uniform(0, 1, n_tracks),
        'brightness': np.random.uniform(0, 1, n_tracks),
        'valence': np.random.uniform(0, 1, n_tracks),
        'speechiness': np.random.uniform(0, 1, n_tracks),
        'acousticness': np.random.uniform(0, 1, n_tracks),
        'instrumentalness': np.random.uniform(0, 1, n_tracks),
        'zero_crossing_rate': np.random.uniform(0, 1, n_tracks),
        'spectral_bandwidth': np.random.uniform(0, 1, n_tracks),
        'harmonic_mean': np.random.uniform(0, 1, n_tracks)
    }

    # Process audio files if they exist
    audio_dir = Path(audio_dir)
    if audio_dir.exists():
        print(f"Processing audio files from {audio_dir}")
        for i, track_id in enumerate(tracks_data['track_id']):
            audio_path = audio_dir / f'{track_id}.mp3'
            if audio_path.exists():
                print(f"Processing track {track_id}")
                features = extract_audio_features(str(audio_path))
                if features:
                    for key, value in features.items():
                        if key in tracks_data:
                            tracks_data[key][i] = value
            else:
                print(f"Audio file not found for {track_id}")
    else:
        print(f"Audio directory {audio_dir} not found. Using random values.")

    users_data = {
        'user_id': [f'user_{i}' for i in range(1, n_users + 1)],
    }

    # Enhanced listening history
    current_date = datetime.datetime.now()
    random_days = [int(x) for x in np.random.randint(0, 365, n_listening_records)]
    dates = [current_date - datetime.timedelta(days=x) for x in random_days]

    listening_data = {
        'user_id': np.random.choice(users_data['user_id'], n_listening_records),
        'track_id': np.random.choice(tracks_data['track_id'], n_listening_records),
        'timestamp': dates,
        'play_count': np.random.randint(1, 11, n_listening_records),
        'time_of_day': [d.strftime('%H:%M') for d in dates],
        'user_mood': np.random.choice(moods, n_listening_records),
    }

    # Create DataFrames and save to CSV
    tracks_df = pd.DataFrame(tracks_data)
    users_df = pd.DataFrame(users_data)
    listening_df = pd.DataFrame(listening_data)

    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    # Save to CSV
    tracks_df.to_csv('output/tracks.csv', index=False)
    users_df.to_csv('output/users.csv', index=False)
    listening_df.to_csv('output/listening_history.csv', index=False)

    return tracks_df, users_df, listening_df

if __name__ == "__main__":
    tracks_df, users_df, listening_df = create_dataset()
    
    print("CSV files have been created successfully!")
    print("\nVerifying created files:")
    print(f"tracks.csv exists: {os.path.exists('output/tracks.csv')}")
    print(f"users.csv exists: {os.path.exists('output/users.csv')}")
    print(f"listening_history.csv exists: {os.path.exists('output/listening_history.csv')}")