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
    ids, X, y = data_loader.load()
    print(ids[0])

    for i in range(len(y[0])):
        if y[0][i] == 1:
            print(i)
    # print(y[0])

