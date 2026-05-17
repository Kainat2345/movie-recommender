import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page title
st.title("🎬 Movie Recommendation System")

# Load dataset
df = pd.read_csv("tmdb_5000_movies.csv")

# Keep important columns only
df = df[['title', 'overview']]

# Remove missing values
df.dropna(inplace=True)

# Convert text into vectors
tfidf = TfidfVectorizer(stop_words='english')

vector = tfidf.fit_transform(df['overview'])

# Similarity matrix
similarity = cosine_similarity(vector)

# Recommendation function
def recommend(movie):
    index = df[df['title'] == movie].index[0]

    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movie_list:
        recommended_movies.append(df.iloc[i[0]].title)

    return recommended_movies

# Dropdown menu
movie_name = st.selectbox(
    "Select a movie",
    df['title'].values
)

# Recommend button
if st.button("Recommend"):

    recommendations = recommend(movie_name)

    st.write("## Recommended Movies")

    for movie in recommendations:
        st.write(movie)