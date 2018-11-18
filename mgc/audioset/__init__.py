import numpy as np
from . import ontology
from .loader import AudiosetDataLoader


MUSIC_GENRE_CLASSES = ontology.find_children('Music genre')


def load_music_genre_instances(datadir):

    print('# Music classes', len(MUSIC_GENRE_CLASSES))

    classes_indexes = [c['index'] for c in MUSIC_GENRE_CLASSES]
    data_loader = AudiosetDataLoader(datadir, classes_indexes)
    ids, X, y = data_loader.load()
    # Redimension 10 secs * 128 features to 1280 features
    X = np.array(X).reshape(-1, 1280)

    return ids, X, y
