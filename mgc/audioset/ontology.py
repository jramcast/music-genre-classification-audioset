import os
import csv
import json

dirname = os.path.dirname(os.path.abspath(__file__))

labels_file_path = os.path.join(dirname, 'class_labels_indices.csv')
ontology_file_path = os.path.join(dirname, 'ontology.json')


def read_classlabels_file():
    with open(labels_file_path, newline='') as csvfile:
        labels_reader = csv.DictReader(csvfile)
        return list(labels_reader)


def read_ontology_file():
    with open(ontology_file_path) as jsonstring:
        return json.load(jsonstring)


classlabels = read_classlabels_file()
ontology = read_ontology_file()


def find_entity_by_name(name):
    return next(entity for entity in ontology if entity['name'] == name)


def find_all_from_name(name):
    entity = find_entity_by_name(name)
    entityid = entity['id']
    return traverse_ontology(entityid)


def find_children(name, drop_parent=True):

    music_classes = find_all_from_name(name)

    # Remove the root class
    if drop_parent:
        music_classes = music_classes[1:]

    # Only select classes present in the dataset (not abstract)
    def add_index_to_class(c):
        c['index'] = get_entity_class_index(c['id'])
        return c
    music_classes = [add_index_to_class(c) for c in music_classes]
    music_classes = [c for c in music_classes if c['index']]
    return music_classes


def traverse_ontology(root_entity_id):
    entity = find_entity(root_entity_id)
    child_ids = entity['child_ids']

    entities = [entity]
    for id in child_ids:
        entities += traverse_ontology(id)

    return entities


def find_entity(entity_id):
    return next(entity for entity in ontology if entity['id'] == entity_id)


def get_entity_class_index(entity_id):
    try:
        classlabel = next(cl for cl in classlabels if cl['mid'] == entity_id)
        index = int(classlabel['index'])
    except StopIteration:
        index = None
    return index
