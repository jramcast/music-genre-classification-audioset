import os
from music_genre.infrastructure.data import AudiosetDataLoader


if __name__ == "__main__":

    datadir = os.path.abspath(
        "./downloads/audioset/audioset_v1_embeddings/bal_train")

    data_loader = AudiosetDataLoader(datadir)
    data_loader.load()
