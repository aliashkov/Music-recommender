import pandas as pd
import numpy as np
import datetime

# Create tracks.csv
np.random.seed(42)
n_tracks = 100

genres = ['Rock', 'Pop', 'Hip Hop', 'Jazz', 'Classical', 'Electronic', 'R&B', 'Metal', 'Folk', 'Latin']
moods = ['Energetic', 'Calm', 'Melancholic', 'Happy', 'Dark', 'Romantic']
decades = ['1970s', '1980s', '1990s', '2000s', '2010s', '2020s']

tracks_data = {
    'track_id': [f'track_{i}' for i in range(1, n_tracks + 1)],
    'title': [f'Song Title {i}' for i in range(1, n_tracks + 1)],
    'artist': [f'Artist {np.random.randint(1, 31)}' for _ in range(n_tracks)],
    'genre': np.random.choice(genres, n_tracks),
    'subgenre': [f'{g} - {np.random.choice(["Type1", "Type2", "Type3"])}' for g in np.random.choice(genres, n_tracks)],
    'mood': np.random.choice(moods, n_tracks),
    'decade': np.random.choice(decades, n_tracks),
    'tempo_bpm': np.random.randint(60, 180, n_tracks),
    'popularity': np.random.randint(1, 101, n_tracks),
    'duration_sec': np.random.randint(120, 500, n_tracks)
}

tracks_df = pd.DataFrame(tracks_data)
tracks_df.to_csv('tracks.csv', index=False)

# Create users.csv
n_users = 50

users_data = {
    'user_id': [f'user_{i}' for i in range(1, n_users + 1)],
    #'age': np.random.randint(16, 70, n_users),
    #'country': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP'], n_users),
    #'activity_level': np.random.choice(['Low', 'Medium', 'High'], n_users)
}

users_df = pd.DataFrame(users_data)
users_df.to_csv('users.csv', index=False)

# Create listening_history.csv
n_listening_records = 1000

current_date = datetime.datetime.now()
# Convert numpy.int32 to regular Python int
random_days = [int(x) for x in np.random.randint(0, 365, n_listening_records)]
dates = [current_date - datetime.timedelta(days=x) for x in random_days]

listening_data = {
    'user_id': np.random.choice(users_data['user_id'], n_listening_records),
    'track_id': np.random.choice(tracks_data['track_id'], n_listening_records),
    'timestamp': dates,
    'play_count': np.random.randint(1, 11, n_listening_records),
    'time_of_day': [d.strftime('%H:%M') for d in dates],
    'user_mood': np.random.choice(moods, n_listening_records)
}

listening_df = pd.DataFrame(listening_data)
listening_df.to_csv('listening_history.csv', index=False)

print("CSV files have been created successfully!")

# Verify files exist
import os
print("\nVerifying created files:")
print(f"tracks.csv exists: {os.path.exists('tracks.csv')}")
print(f"users.csv exists: {os.path.exists('users.csv')}")
print(f"listening_history.csv exists: {os.path.exists('listening_history.csv')}")