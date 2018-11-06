
from music_genre.infrastructure.data import AudiosetDataLoader


class BayesClassifier:

    def run(self):

        data_loader = AudiosetDataLoader()
        data_loader.load()
        print("Naive Bayes experiment")
