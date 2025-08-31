Movie Recommendation System

A simple content-based movie recommendation system built with Python.
It recommends movies based on similar genres and overviews using TF-IDF (Term Frequency–Inverse Document Frequency) and Cosine Similarity.

✨ Features

📜 Built-in dataset of 50 movies (title, genre, overview)

🔎 User selects a movie → system suggests 5 similar movies

🧠 Uses TF-IDF Vectorization for feature extraction

📐 Computes Cosine Similarity to find closeness between movies

🎯 Fast and lightweight — no external dataset required

🛠️ Tech Stack

Python 3

Pandas for dataset handling

Scikit-learn (TfidfVectorizer, cosine_similarity) for ML features

🚀 How to Run

Save the code in a file named movies_recommendation.py

Open a terminal in the same folder

Run the script:

python movies_recommendation.py

📌 Example Usage
🎬 Welcome to the Movie Recommendation System 🎬

Here are some movies to choose from:

1. The Matrix
2. The Matrix Reloaded
3. John Wick
4. John Wick 2
5. John Wick 3
...
50. Toy Story

Pick a movie by number: 1

Because you selected 'The Matrix', you may also like:

👉 The Matrix Reloaded
👉 Inception
👉 Tenet
👉 Interstellar
👉 The Dark Knight

📚 Future Improvements

Add a larger dataset from IMDb or TMDB

Use user ratings for better recommendations

Add a GUI/web interface

Implement hybrid (content + collaborative) filtering

👨‍💻 Author

Created by [Your Name] using Python 🐍 and scikit-learn 📊
