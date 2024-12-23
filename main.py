from music_recommender import EnhancedMusicRecommender
from create_dataset import create_dataset
import os

def main():
    # Create dataset if it doesn't exist
    if not all(os.path.exists(f) for f in ['tracks.csv', 'users.csv', 'listening_history.csv']):
        print("Creating new dataset...")
        create_dataset()
    
    # Initialize recommender
    recommender = EnhancedMusicRecommender()
    
    # Get recommendations for a user
    user_id = "user_2"
    recommendations = recommender.get_recommendations(
        user_id,
        n_recommendations=5,
        consider_recent_mood=True
    )
    
    # Print user's recent mood
    recent_mood = recommender.get_recent_mood(user_id)
    print(f"\nUser's recent mood: {recent_mood}")
    
    # Print recommendations
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"\nTrack: {rec['title']}")
        print(f"Artist: {rec['artist']}")
        print(f"Genre: {rec['genre']}")
        print(f"Mood: {rec['mood']}")
        print(f"Danceability: {rec['danceability']:.2f}")
        print(f"Energy: {rec['energy']:.2f}")
        print(f"Score: {rec['score']:.2f}")

if __name__ == "__main__":
    main()