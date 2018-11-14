import numpy as np
from . import ontology, transform
from .loader import AudiosetDataLoader


MUSIC_GENRE_CLASSES = ontology.find_children('Music genre')


def load_music_genre_instances(datadir):

    print('# Music classes', len(MUSIC_GENRE_CLASSES))

    data_loader = AudiosetDataLoader(datadir)
    ids, X, y = data_loader.load()
    # Redimension 10 secs * 128 features to 1280 features
    X = np.array(X).reshape(-1, 1280)

    # Filter only data targeted as music
    X, y = transform.subset_by_class(X, y, MUSIC_GENRE_CLASSES)

    return X, y
