import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Sample Dataset creation (Simulating a database of movies)
movies_data = {
    "movie_id": [0, 1, 2, 3, 4, 5],
    "title": [
        "The Dark Knight",
        "Inception",
        "Interstellar",
        "The Matrix",
        "The Notebook",
        "La La Land",
    ],
    "genres": [
        "Action Action Thriller",
        "Action Sci-Fi Thriller",
        "Sci-Fi Adventure Drama",
        "Action Sci-Fi",
        "Romance Drama",
        "Romance Musical Comedy",
    ],
    "director": [
        "Christopher Nolan",
        "Christopher Nolan",
        "Christopher Nolan",
        "Lana Wachowski",
        "Nick Cassavetes",
        "Damien Chazelle",
    ],
    "keywords": [
        "batman superhero dc grim",
        "dreams heist mind-bending",
        "space exploration time-dilation",
        "simulation cyber artificial-intelligence",
        "love-story heartbreak relationship",
        "jazz dancing dreams love-story",
    ],
}

# Load data into a Pandas DataFrame
df = pd.DataFrame(movies_data)


def combine_features(row):
    """Combines metadata features into a single metadata soup string."""
    # We lowercase everything to ensure match uniformity
    return f"{row['genres']} {row['director']} {row['keywords']}".lower()


# Create a unified feature column
df["combined_features"] = df.apply(combine_features, axis=1)

# 2. Vectorization & Similarity Matrix Matrix Calculation
# Convert text strings into word count frequency vectors
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

# Compute the pairwise cosine similarity matrix across all movies
cosine_sim = cosine_similarity(count_matrix)


def get_recommendations(movie_title, similarity_matrix, dataframe, top_n=2):
    """Recommends movies similar to the given movie title."""
    try:
        # Find the index corresponding to the given movie title
        movie_index = dataframe[dataframe["title"].str.lower() == movie_title.lower()].index[0]
    except IndexError:
        return f"Error: '{movie_title}' was not found in the database. Please try another film."

    # Fetch row similarity scores for that specific movie index
    # Enumerate helps map score back to the original database row index -> (index, similarity_score)
    similar_movies = list(enumerate(similarity_matrix[movie_index]))

    # Sort the list based on similarity scores in descending order
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

    print(f"\nBecause you watched '{dataframe.iloc[movie_index]['title']}':")
    
    # Track output count (skip index 0 because it's the movie itself)
    count = 0
    for element in sorted_similar_movies:
        idx = element[0]
        score = element[1]
        
        if idx == movie_index:
            continue
            
        print(f" -> Recommended: {dataframe.iloc[idx]['title']} (Match Confidence: {score*100:.1f}%)")
        count += 1
        if count >= top_n:
            break


def main():
    print("====================================================")
    print("        Content-Based Movie Recommendation Engine   ")
    print("====================================================")
    
    print("Available movies in database:")
    for t in df["title"].tolist():
        print(f" • {t}")
    print("-" * 50)

    while True:
        user_choice = input("\nEnter a movie title from above for recommendations (or 'exit' to quit): ").strip()
        
        if user_choice.lower() in ["exit", "quit", "q"]:
            print("Exiting Recommendation System. Goodbye!")
            break
            
        if not user_choice:
            continue
            
        get_recommendations(user_choice, cosine_sim, df, top_n=2)


if __name__ == "__main__":
    main()