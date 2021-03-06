import os
import csv
from mgc.audioset import ontology
from mgc.audioset.loaders import MusicGenreSubsetLoader
from mgc.audioset.transform import tensor_to_numpy


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), './data/'))
TF_RECORDS_DIR = os.path.join(DATA_DIR, 'bal_train')
MUSIC_GENRE_CLASSES = ontology.find_children('Music genre')
MUSIC_GENRE_CLASSES_BY_ID = {}
for c in MUSIC_GENRE_CLASSES:
    MUSIC_GENRE_CLASSES_BY_ID[c['id']] = c


def test_bal_has_expected_size():
    '''
    This test check if we are reading tfrecords correctly
    in TF tensor format
    by comparing the read data with the Audioset's csv format.
    This test uses the balanced dataset
    '''
    loader = MusicGenreSubsetLoader(DATA_DIR, repeat=False)
    ids, X, y = loader.load_bal()
    ids, X, y = tensor_to_numpy(ids, X, y)
    assert X.shape == (2490, 10, 128)
    assert y.shape == (2490, 53)


def test_eval_has_expected_size():
    '''
    This test check if we are reading tfrecords correctly
    in TF tensor format
    by comparing the read data with the Audioset's csv format.
    This test uses the balanced dataset
    '''
    loader = MusicGenreSubsetLoader(DATA_DIR, repeat=False)
    ids, X, y = loader.load_eval()
    ids, X, y = tensor_to_numpy(ids, X, y)
    assert X.shape == (2048, 10, 128)
    assert y.shape == (2048, 53)


# def test_load_music_genre_subset_as_tensor_match_csv_samples():
#     '''
#     This test check if we are reading tfrecords correctly
#     in TF tensor format
#     by comparing the read data with the Audioset's csv format.
#     This test uses the balanced dataset
#     '''
#     csv_samples = read_audioset_balanced_csv()
#     loader = MusicGenreSubsetLoader(DATA_DIR, repeat=False)
#     ids, X, y = loader.load_bal()
#     ids, X, y = tensor_to_numpy(ids, X, y)

#     for i in range(len(ids)):
#         video_id = ids[i].decode('utf-8')
#         csv_sample = csv_samples[video_id]
#         csv_label_ids = csv_sample['positive_labels']
#         assert_classes_in_csv_sample_match_tf_sample(csv_label_ids, y[i])


# def assert_classes_in_csv_sample_match_tf_sample(csv_label_ids, y_row):
#     # Find active classes: Get ids of columns containing 1's
#     classes_indexes = np.argwhere(y_row)[0].tolist()
#     classes_ids = [MUSIC_GENRE_CLASSES[i]['id'] for i in classes_indexes]
#     assert set(classes_ids).issubset(csv_label_ids)


def read_audioset_balanced_csv():
    filepath = os.path.join(DATA_DIR, 'balanced_train_segments.csv')
    dataset = {}
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)
        # skip first 3 lines
        next(reader)
        next(reader)
        next(reader)
        for row in reader:
            sample = {
                'YTID': row[0],
                'start_seconds': row[1],
                'end_seconds': row[2],
                'positive_labels': row[3].split(',')
            }
            dataset[sample['YTID']] = sample

        return dataset
