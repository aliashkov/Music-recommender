from music_recommender import EnhancedMusicRecommender
from create_dataset import create_dataset
import os

def main():
    # Create dataset if it doesn't exist
    #if not all(os.path.exists(os.path.join('output', f)) for f in ['tracks.csv', 'users.csv', 'listening_history.csv']):
    #    print("Creating new dataset...")
    #    create_dataset()
    
    try:
        # Initialize recommender
        recommender = EnhancedMusicRecommender()
        
        # Get recommendations for a user
        user_id = "user_20"
        recommendations = recommender.get_recommendations(
            user_id,
            n_recommendations=5,
            consider_recent_mood=False
        )
        
        # Print user's recent mood
        recent_mood = recommender.get_recent_mood(user_id)
        print(f"\nUser's recent mood: {recent_mood}")
        
        # Print recommendations
        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"\nTrack: {rec['title']}")
                print(f"Artist: {rec['artist']}")
                print(f"Genre: {rec['genre']}")
                print(f"Mood: {rec['mood']}")
                print(f"Danceability: {rec['danceability']:.2f}")
                print(f"Energy: {rec['energy']:.2f}")
                print(f"Score: {rec['score']:.2f}")
        else:
            print("\nNo recommendations found for this user.")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()