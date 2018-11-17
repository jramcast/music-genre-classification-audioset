import numpy as np
from mgc.audioset import transform, ontology


MUSIC_GENRE_CLASSES = ontology.find_children('Music genre')


def test_take_y_for_classes_returns_a_matrix_only_for_the_requested_classes():

    # GIVEN 3 samples with 527 possible labels each
    num_samples = 3
    num_classes = 527
    y = np.full((num_samples, num_classes), 0)
    # We initialize the values of the 3 samples for some
    # random classes to later check that they are in the right place
    class_216 = [0, 0, 1]
    class_231 = [0, 1, 0]
    class_237 = [0, 1, 1]
    class_265 = [1, 0, 0]
    class_30 = [1, 0, 1]
    y[:, 216] = class_216
    y[:, 231] = class_231
    y[:, 237] = class_237
    y[:, 265] = class_265
    y[:, 30] = class_30

    # GIVEN only music genre labels are taken from the 527
    y = transform.take_y_for_classes(y, MUSIC_GENRE_CLASSES)

    # THEN we expect to have only the selected classes
    assert y.shape == (3, 53)


def test_take_y_for_classes_makes_the_right_selection():

    # GIVEN 3 samples with 527 possible labels each
    num_samples = 3
    num_classes = 527
    y = np.full((num_samples, num_classes), 0)
    # We initialize the values of the 3 samples for some
    # random classes to later check that they are in the right place
    class_216 = [0, 0, 1]
    class_231 = [0, 1, 0]
    class_237 = [0, 1, 1]
    class_265 = [1, 0, 0]
    class_30 = [1, 0, 1]
    y[:, 216] = class_216
    y[:, 231] = class_231
    y[:, 237] = class_237
    y[:, 265] = class_265
    y[:, 30] = class_30

    # GIVEN only music genre labels are taken from the 527
    y = transform.take_y_for_classes(y, MUSIC_GENRE_CLASSES)

    # THEN we expect to see the selected classes in the exact same positions
    # that they have in the MUSIC_GENRE_CLASSES list
    assert np.array_equal(y[:, 0], class_216)
    assert np.array_equal(y[:, 40], class_30)
    assert np.array_equal(y[:, 15], class_231)
    assert np.array_equal(y[:, 21], class_237)
    assert np.array_equal(y[:, 52], class_265)
