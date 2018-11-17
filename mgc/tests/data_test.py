from mgc.audioset import load_music_genre_instances, ontology


def test_data_loaded_ok():

    load_music_genre_instances("mgc/tests/data/bal_train")