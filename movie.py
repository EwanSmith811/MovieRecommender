from flask import Flask, jsonify, request, render_template
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

movies = pd.read_csv(r"C:\Users\thech\Downloads\Movie\Data\movies.csv")
ratings = pd.read_csv(r"C:\Users\thech\Downloads\Movie\Data\ratings.csv")

def cleanTitle(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

movies["clean_title"] = movies["title"].apply(cleanTitle)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])
def searchTitle(title):
    title = cleanTitle(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -10)[-10:]
    results = movies.iloc[indices][::-1]
    results = results[similarity[indices] > 0.1]
    return results

def getRecommendations(movieId, genre=None):
    similarUsers = ratings[(ratings["movieId"] == movieId) & (ratings["rating"] >= 4)]["userId"].unique()
    recs = ratings[(ratings["userId"].isin(similarUsers)) & (ratings["rating"] >= 4)]["movieId"]
    recs = recs.value_counts() / len(similarUsers)
    recs = recs[recs > .1]
    allUsers = ratings[(ratings["movieId"].isin(recs.index)) & (ratings["rating"] >= 4)]
    allRecs = allUsers["movieId"].value_counts() / len(allUsers["userId"].unique())
    allRecs = allRecs[allRecs > .1]
    recPercent = pd.concat([recs, allRecs], axis=1)
    recPercent.columns = ["similar", "all"]
    recPercent["score"] = recPercent["similar"] / recPercent["all"]
    recPercent = recPercent.sort_values("score", ascending=False)
    recommendations = recPercent.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]
    if genre:
        recommendations = recommendations[recommendations["genres"].str.contains(genre, case=False, na=False)]
    if recommendations.empty:
        return "No Movies Found that Fit this Description"
    return recommendations

@app.route('/recommendations', methods=["POST"])
def getRecommendationsByTitle():
    data = request.get_json()
    movieTitle = data.get('title', '')
    genre = data.get('genre', None)
    searchResults = searchTitle(movieTitle)
    if not searchResults.empty:
        movieId = searchResults.iloc[0]["movieId"]
        recommendations = getRecommendations(movieId, genre)
        return jsonify(recommendations.to_dict(orient='records'))
    else:
        return jsonify([])

@app.route('/')
def home():
    return render_template('movie.html')

if __name__ == "__main__":
    app.run(debug=True)
