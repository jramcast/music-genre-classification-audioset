import logging
import numpy as np
from sklearn import metrics
from scipy.stats import norm


def calculate_stats(output, y):
    """Calculate statistics for each class

    Args:
      output: 2d array, (samples_num, classes_num)
      y: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistics of each class.
    """

    classes_num = y.shape[-1]
    stats = []

    # Class-wise statistics
    for i in range(classes_num):
        class_stats = calculate_class_stats(y, i, output)
        stats.append(class_stats)

    return stats


def calculate_class_stats(y, class_index, output):
    # Average precision
    avg_precision = metrics.average_precision_score(
        y[:, class_index], output[:, class_index], average=None)

    # AUC
    if y.shape[1] > 1:
        auc = metrics.roc_auc_score(
            y[:, class_index], output[:, class_index], average=None)
    else:
        auc = 0

    # precission and recall
    precision = metrics.precision_score(
        y[:, class_index], output[:, class_index] >= 0.5)
    recall = metrics.recall_score(
        y[:, class_index], output[:, class_index] >= 0.5)
    f1 = metrics.f1_score(y[:, class_index], output[:, class_index] >= 0.5)

    # confusion matrix
    tn, fp, fn, tp = metrics.confusion_matrix(
        y[:, class_index], output[:, class_index] >= 0.5).ravel()

    class_stats = {
        'AP': avg_precision,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
    }
    return class_stats


def d_prime(auc):
    standard_normal = norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime


def log_metrics_per_class(stats, music_classes, sortkey='AP', top=10):

    music_classes_stats = []
    for index, class_info in enumerate(music_classes):
        # class_index = audioset.utils.get_entity_class_index(class_info['id'])

        music_stats = dict(stats[index])
        music_stats['name'] = class_info['name']
        music_classes_stats.append(music_stats)

    best_music_classes_stats = sorted(
        music_classes_stats, key=lambda k: -k[sortkey])
    worst_music_classes_stats = sorted(
        music_classes_stats, key=lambda k: k[sortkey])

    # logging.info best
    logging.info("  Best music classes:")
    for class_info in best_music_classes_stats[:top]:
        logging.info("      {}  -  map: {:.6f}, auc: {:.6f}, precision: {:.6f}, recall: {:.6f}, f1: {:.6f}, tp: {:.6f}, fp: {:.6f}, tn: {:.6f}, fn: {:.6f}".format(
            class_info['name'],
            class_info['AP'],
            class_info['auc'],
            class_info['precision'],
            class_info['recall'],
            class_info['f1'],
            class_info['tp'],
            class_info['fp'],
            class_info['tn'],
            class_info['fn'],
        ))

    # logging.info worst
    logging.info("  Worst music classes:")
    for class_info in worst_music_classes_stats[:top]:
        logging.info("      {}  -  map: {:.6f}, auc: {:.6f}, precision: {:.6f}, recall: {:.6f}, f1: {:.6f}, tp: {:.6f}, fp: {:.6f}, tn: {:.6f}, fn: {:.6f}".format(
            class_info['name'],
            class_info['AP'],
            class_info['auc'],
            class_info['precision'],
            class_info['recall'],
            class_info['f1'],
            class_info['tp'],
            class_info['fp'],
            class_info['tn'],
            class_info['fn'],
        ))


def get_avg_stats(output, target, classes=None, num_classes=10):
    """Average predictions of different iterations and compute stats
    """

    # Calculate stats
    stats = calculate_stats(output, target)

    # Write out to log
    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])

    m_precision = np.mean([stat['precision'] for stat in stats])
    m_recall = np.mean([stat['recall'] for stat in stats])
    f1 = np.mean([stat['f1'] for stat in stats])
    dprime = d_prime(mAUC)

    if classes:
        log_metrics_per_class(stats, classes, top=num_classes)

    logging.info("mAP: {:.6f}".format(mAP))
    logging.info("AUC: {:.6f}".format(mAUC))
    logging.info("d_prime: {:.6f}".format(dprime))
    logging.info("mPrecision: {:.6f}".format(m_precision))
    logging.info("mRecall: {:.6f}".format(m_recall))
    logging.info("mf1: {:.6f}".format(f1))

    return mAP, mAUC, d_prime(mAUC)
