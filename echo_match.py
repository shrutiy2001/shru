import os
import requests
import re
from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "cleaned_dataset.csv")
songs_df = pd.read_csv(DATA_PATH)

audio_features = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms"
]

X = songs_df[audio_features].fillna(0)
y = songs_df["track_genre"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=50, random_state=42)
)
pipeline.fit(X_train, y_train)


def calculate_mood(row):
    mood = []
    if row["valence"] > 0.6: mood.append("Happy")
    if row["valence"] < 0.3: mood.append("Sad")
    if row["energy"] > 0.7: mood.append("Energetic")
    if row["danceability"] > 0.7: mood.append("Dance")
    return ", ".join(mood) if mood else "Neutral"


def search_songs(query):
    q = query.lower()
    return songs_df[
        songs_df["track_name"].str.lower().str.contains(q, na=False) |
        songs_df["artists"].str.lower().str.contains(q, na=False)
    ]


# ⭐ Fetch real YouTube video
def get_youtube_video(query):
    url = "https://www.youtube.com/results?search_query=" + query
    html = requests.get(url).text

    match = re.search(r"watch\?v=([A-Za-z0-9_-]{11})", html)
    if match:
        return "https://www.youtube.com/embed/" + match.group(1)
    return ""


@app.route("/")
@app.route("/search")
def search_page():
    return render_template("search.html")

@app.route("/mood")
def mood_page():
    return render_template("mood.html")

@app.route("/results")
def results_page():
    query = request.args.get("q", "").strip()

    if not query:
        return render_template("results.html", songs=[], query="", count=0)

    matches = search_songs(query).head(20)

    songs = []
    for _, row in matches.iterrows():
        yt_query = f"{row['track_name']} {row['artists']}".replace(" ", "+")
        youtube_link = youtube_link = f"https://www.youtube.com/results?search_query={yt_query}"
        songs.append({
            "track_name": row["track_name"],
            "artist": row["artists"],
            "album": row["album_name"],
            "predicted": row["track_genre"],
            "confidence": 100,
            "mood": calculate_mood(row),
            "image_url": "/static/placeholder_vinyl.svg",
            "youtube": youtube_link
        })

    return render_template("results.html", songs=songs, query=query, count=len(songs))


if __name__ == "__main__":
    app.run(debug=True)