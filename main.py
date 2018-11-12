import os
import time
import numpy as np
from music_genre.infrastructure.data import AudiosetDataLoader
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB


if __name__ == "__main__":
    datadir = os.environ.get(
        "DATA_DIR",
        "./downloads/audioset/audioset_v1_embeddings/bal_train"
    )
    datadir = os.path.abspath(datadir)
    print(datadir)

    data_loader = AudiosetDataLoader(datadir)
    ids, X, y = data_loader.load()
    # Redimension 10 secs * 128 features to 1280 features
    X = np.array(X).reshape(-1, 1280)


    X_train, X_validate, y_train, y_validate = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    print(X_train.shape)
    print(y_train.shape)

    classifier = OneVsRestClassifier(GaussianNB(), n_jobs=-1)

    print("Training {}...".format(classifier))
    start_time = time.time()

    classifier.fit(X_train, y_train)
    print("Training data time: {:.3f} s".format(time.time() - start_time))



