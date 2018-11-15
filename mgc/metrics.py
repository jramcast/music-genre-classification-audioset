import numpy as np
from sklearn import metrics
from scipy.stats import norm


def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        if target.shape[1] > 1:
            auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)
        else:
            auc = 0

        # accuracy
        accuracy = metrics.accuracy_score(target[:, k], output[:, k] > 0.5)

        # precission and recall
        precision = metrics.precision_score(target[:, k], output[:, k] > 0.5)
        recall = metrics.recall_score(target[:, k], output[:, k] > 0.5)
        f1 = metrics.f1_score(target[:, k], output[:, k] > 0.5)

        # confusion matrix
        tn, fp, fn, tp = metrics.confusion_matrix(target[:, k], output[:, k] > 0.5).ravel()

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, k], output[:, k])

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        save_every_steps = 1000     # Sample statistics to reduce size
        dict = {
            'precisions': precisions[0::save_every_steps],
            'recalls': recalls[0::save_every_steps],
            'AP': avg_precision,
            'fpr': fpr[0::save_every_steps],
            'fnr': 1. - tpr[0::save_every_steps],
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
        }

        stats.append(dict)

    return stats


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

    best_music_classes_stats = sorted(music_classes_stats, key=lambda k: -k[sortkey])
    worst_music_classes_stats = sorted(music_classes_stats, key=lambda k: k[sortkey])

    # print best
    print("  Best music classes:")
    for class_info in best_music_classes_stats[:top]:
        print("      {}  -  map: {:.6f}, auc: {:.6f}, precision: {:.6f}, recall: {:.6f}, accuracy: {:.6f},  f1: {:.6f}, tp: {:.6f}, fp: {:.6f}, tn: {:.6f}, fn: {:.6f}".format(
            class_info['name'],
            class_info['AP'],
            class_info['auc'],
            class_info['precision'],
            class_info['recall'],
            class_info['accuracy'],
            class_info['f1'],
            class_info['tp'],
            class_info['fp'],
            class_info['tn'],
            class_info['fn'],
        ))

    # print worst
    print("  Worst music classes:")
    for class_info in worst_music_classes_stats[:top]:
        print("      {}  -  map: {:.6f}, auc: {:.6f}, precision: {:.6f}, recall: {:.6f}, accuracy: {:.6f},  f1: {:.6f}, tp: {:.6f}, fp: {:.6f}, tn: {:.6f}, fn: {:.6f}".format(
            class_info['name'],
            class_info['AP'],
            class_info['auc'],
            class_info['precision'],
            class_info['recall'],
            class_info['accuracy'],
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
    m_accuracy = np.mean([stat['accuracy'] for stat in stats])
    f1 = np.mean([stat['f1'] for stat in stats])
    dprime = d_prime(mAUC)

    if classes:
        log_metrics_per_class(stats, classes, top=num_classes)

    print("mAP: {:.6f}".format(mAP))
    print("AUC: {:.6f}".format(mAUC))
    print("d_prime: {:.6f}".format(dprime))
    print("mPrecision: {:.6f}".format(m_precision))
    print("mRecall: {:.6f}".format(m_recall))
    print("mf1: {:.6f}".format(f1))

    return mAP, mAUC, d_prime(mAUC)
