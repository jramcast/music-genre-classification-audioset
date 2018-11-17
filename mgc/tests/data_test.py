import os
import csv
from mgc.audioset import load_music_genre_instances, ontology


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), './data/'))
TF_RECORDS_DIR = os.path.join(DATA_DIR, 'bal_train')
MUSIC_GENRE_CLASSES = ontology.find_children('Music genre')
MUSIC_GENRE_CLASSES_BY_ID = {}
for c in MUSIC_GENRE_CLASSES:
    MUSIC_GENRE_CLASSES_BY_ID[c['id']] = c


def test_audioset_loaded_tfrecords_match_csv_samples():
    '''
    This test check if we are reading tfrecords correctly
    by comparing the read data with the Audioset's csv format.
    The tests uses the balanced dataset
    '''
    read_audioset_balanced_csv()
    csv_samples = read_audioset_balanced_csv()
    tf_samples = load_music_genre_instances(TF_RECORDS_DIR)
    ids, X, y = tf_samples

    for i in range(len(ids)):
        video_id = ids[i].decode('utf-8')
        csv_sample = csv_samples[video_id]
        csv_label_ids = csv_sample['positive_labels']
        assert_classes_in_csv_sample_match_tf_sample(csv_label_ids, y[i])


def assert_classes_in_csv_sample_match_tf_sample(csv_label_ids, y_row):
    for class_id in csv_label_ids:
        if class_id in MUSIC_GENRE_CLASSES_BY_ID:
            class_index = MUSIC_GENRE_CLASSES_BY_ID[class_id]['index']
            assert y_row[class_index] == 1


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
