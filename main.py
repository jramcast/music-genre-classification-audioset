import os
from music_genre.infrastructure.data import AudiosetDataLoader


if __name__ == "__main__":
    datadir = os.environ.get(
        "DATA_DIR",
        "./downloads/audioset/audioset_v1_embeddings/bal_train"
    )
    datadir = os.path.abspath(datadir)
    print(datadir)

    data_loader = AudiosetDataLoader(datadir)
    data_loader.load()
