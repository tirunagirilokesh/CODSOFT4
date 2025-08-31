import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 50 Movies dataset
movies = {
    'title': [
        'The Matrix', 'The Matrix Reloaded', 'John Wick', 'John Wick 2', 'John Wick 3',
        'Inception', 'Interstellar', 'The Dark Knight', 'The Dark Knight Rises', 'Memento',
        'Tenet', 'Avatar', 'Titanic', 'Aliens', 'Terminator 2',
        'Iron Man', 'Iron Man 2', 'Iron Man 3', 'The Avengers', 'Avengers: Endgame',
        'Guardians of the Galaxy', 'Guardians Vol. 2', 'Thor', 'Thor: Ragnarok', 'Black Panther',
        'Doctor Strange', 'Spider-Man: Homecoming', 'Spider-Man: No Way Home', 'Logan', 'Deadpool',
        'The Shawshank Redemption', 'The Godfather', 'The Godfather Part II', 'Pulp Fiction', 'Fight Club',
        'Se7en', 'Whiplash', 'La La Land', 'Parasite', 'The Social Network',
        'Dune', 'Arrival', 'Blade Runner 2049', 'Joker', 'The Batman',
        'Frozen', 'Moana', 'Zootopia', 'Coco', 'Toy Story'
    ],
    'genre': [
        'Sci-Fi Action','Sci-Fi Action','Action Thriller','Action Thriller','Action Thriller',
        'Sci-Fi Thriller','Sci-Fi Drama','Action Crime','Action Crime','Mystery Thriller',
        'Sci-Fi Action','Sci-Fi Adventure','Romance Drama','Sci-Fi Action','Sci-Fi Action',
        'Superhero Action','Superhero Action','Superhero Action','Superhero Action','Superhero Action',
        'Superhero Comedy','Superhero Comedy','Superhero Fantasy','Superhero Comedy','Superhero Action',
        'Superhero Fantasy','Superhero Action','Superhero Action','Superhero Drama','Superhero Comedy',
        'Drama','Crime Drama','Crime Drama','Crime Drama','Drama Thriller',
        'Crime Thriller','Drama Music','Romance Music','Thriller Drama','Drama',
        'Sci-Fi Adventure','Sci-Fi Drama','Sci-Fi Drama','Crime Drama','Crime Action',
        'Animation Fantasy','Animation Adventure','Animation Comedy','Animation Drama','Animation Comedy'
    ],
    'overview': [
        'A hacker discovers reality is a simulation and joins the rebellion.',
        'Neo and rebels fight against the Machines in the Matrix.',
        'A retired hitman seeks vengeance for his dog.',
        'John Wick returns to the criminal underworld.',
        'John Wick faces his toughest challenge yet.',
        'A thief enters dreams to steal secrets and must plant an idea.',
        'Explorers travel through a wormhole to save humanity.',
        'Batman faces the Joker who brings chaos to Gotham.',
        'Batman returns to fight the merciless Bane.',
        'A man with memory loss hunts for his wife’s killer.',
        'An agent manipulates time inversion to prevent World War III.',
        'A Marine discovers Pandora and its alien world.',
        'A romance unfolds on the ill-fated Titanic.',
        'Ripley returns to fight deadly aliens.',
        'A cyborg protects John Connor from a killer robot.',
        'Tony Stark builds a suit and becomes Iron Man.',
        'Tony Stark battles a powerful enemy.',
        'Tony Stark faces the Mandarin.',
        'Earth’s heroes unite against an alien invasion.',
        'The Avengers fight their final battle against Thanos.',
        'Misfit heroes save the galaxy.',
        'The Guardians return to protect the universe again.',
        'Thor must defend Asgard from his brother Loki.',
        'Thor faces Hela to save Asgard.',
        'T’Challa returns to Wakanda to claim his throne.',
        'A surgeon discovers mystic arts after an accident.',
        'Peter Parker balances high school and being Spider-Man.',
        'Spider-Man fights villains across the multiverse.',
        'An aging Wolverine protects a young mutant.',
        'A merc with a mouth seeks revenge.',
        'Two prisoners bond and find redemption.',
        'The patriarch of a crime dynasty hands power to his son.',
        'Michael Corleone expands the crime empire.',
        'Hitmen, a boxer, and criminals intertwine in L.A.',
        'A man forms a secret fight club.',
        'Two detectives hunt a killer using the seven sins.',
        'A drummer faces his ruthless teacher.',
        'A jazz musician and actress fall in love.',
        'A poor family schemes to live with a rich family.',
        'The rise of Facebook and its lawsuits.',
        'A gifted man must save his desert planet.',
        'Aliens visit Earth and a linguist tries to communicate.',
        'A young man finds his destiny in a dystopian future.',
        'A failed comedian becomes the Joker.',
        'Batman faces the Riddler in Gotham.',
        'Two sisters discover magical powers in Arendelle.',
        'A young girl sails the ocean to save her island.',
        'Animals live in a city of harmony.',
        'A boy discovers family secrets through music.',
        'Toys come alive when humans aren’t looking.'
    ]
}

# Create DataFrame
df = pd.DataFrame(movies)

# Combine genre + overview for features
df['content'] = df['genre'] + " " + df['overview']

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['content'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_movie(choice_index, df, cosine_sim, top_n=5):
    sim_scores = list(enumerate(cosine_sim[choice_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    rec_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[rec_indices].tolist()

# --- Main Flow ---
print("🎬 Welcome to the Movie Recommendation System 🎬\n")
print("Here are some movies to choose from:\n")

for i, title in enumerate(df['title'], 1):
    print(f"{i}. {title}")

choice = int(input("\nPick a movie by number: ")) - 1

if 0 <= choice < len(df):
    selected = df['title'][choice]
    print(f"\nBecause you selected '{selected}', you may also like:\n")
    recs = recommend_movie(choice, df, cosine_sim, top_n=5)
    for rec in recs:
        print("👉", rec)
else:
    print("❌ Invalid choice. Please run again.")
