import numpy as np
from . import ontology
from .loaders import NPArrayLoader, TensorLoader


MUSIC_GENRE_CLASSES = ontology.find_children('Music genre')


def load_music_genre_subset_as_numpy(datadir):
    classes_indexes = [c['index'] for c in MUSIC_GENRE_CLASSES]
    data_loader = NPArrayLoader(datadir, classes_indexes)
    ids, X, y = data_loader.load()
    # Redimension 10 secs * 128 features to 1280 features
    X = np.array(X).reshape(-1, 1280)
    return ids, X, y


def load_music_genre_subset_as_tensor(datadir, repeat=True):
    classes_indexes = [c['index'] for c in MUSIC_GENRE_CLASSES]
    data_loader = TensorLoader(datadir, classes_indexes, repeat)
    return data_loader.load()
