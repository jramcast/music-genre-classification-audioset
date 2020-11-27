import os
import sys
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../.."
    )
)
import json

from mgc.audioset import ontology


predictions = [
    ("Chant", 0.9),
    ("A capella", 0.7),
    ("Mantra", 0.6),
    ("Blues", 0.5),
    ("Classical music", 0.4),
    ("Opera", 0.3),
    ("Vocal music", 0.3)
]


# https://towardsdatascience.com/how-to-build-a-fast-most-similar-words-method-in-spacy-32ed104fe498

def compare_genres(a, b):
    a = preprocess(a)
    b = preprocess(b)

    if a == b:
        return 1
    if a in b or b in a:
        return 0.5

    return 0

    # genres = ontology.find_all_from_name(a)

def preprocess(text: str):
    return text.strip().lower()

if __name__ == "__main__":

    with open('scripts/lastfm/song_tags.json') as json_file:
        data = json.load(json_file)

    lastfm_genres = data["Queen - Bohemian Rhapsody"]

    for genre in lastfm_genres:
        for prediction in predictions:
            result = compare_genres(genre[0], prediction[0])

            if result > 0:
                print(genre[0], prediction[0], result)

# for genre in ontology.find_all_from_name("Music genre"):
#     print(genre)