import os
from music_genre.infrastructure.data import AudiosetDataLoader
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    datadir = os.environ.get(
        "DATA_DIR",
        "./downloads/audioset/audioset_v1_embeddings/bal_train"
    )
    datadir = os.path.abspath(datadir)
    print(datadir)

    data_loader = AudiosetDataLoader(datadir)
    for id in data_loader.load():
        print('ID', id)

