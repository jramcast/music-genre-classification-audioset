import logging
import csv
import numpy as np
from collections import namedtuple
from sklearn import metrics
from scipy.stats import norm
from typing import List


ClassMetrics = namedtuple('ClassMetrics', [
    'name',
    'ap',
    'auc',
    'precision',
    'recall',
    'f1',
    'tp',
    'fp',
    'tn',
    'fn'
])


class MetricsLogger:

    def __init__(self,
                 classes=False,
                 classsmetrics_filepath=None,
                 show_top_classes=10,
                 class_sort_key='ap'):
        self.classes = classes
        self.classsmetrics_filepath = classsmetrics_filepath
        self.show_top_classes = show_top_classes
        self.class_sort_key = class_sort_key
        # Metrics
        self.ap = None
        self.auc = None
        self.dprime = None
        self.class_metrics = []

    def log(self, y_predicted: np.array, y_true: np.array, show_classes=False):
        # Calculate metrics for each class
        metrics = self.calculate_metrics(y_predicted, y_true)

        # Log average metrics among all classes
        m_ap = np.mean([classmetrics.ap for classmetrics in metrics])
        m_auc = np.mean([classmetrics.auc for classmetrics in metrics])
        m_precision = np.mean([classmetrics.precision for classmetrics in metrics])
        m_recall = np.mean([classmetrics.recall for classmetrics in metrics])
        f1 = np.mean([classmetrics.f1 for classmetrics in metrics])
        dprime = self.calculate_dprime(m_auc)

        if show_classes:
            self.log_classmetrics(metrics)
            self.save_classmetrics(metrics)

        logging.info("mAP: {:.6f}".format(m_ap))
        logging.info("mAUC: {:.6f}".format(m_auc))
        logging.info("d_prime: {:.6f}".format(dprime))
        logging.info("mPrecision: {:.6f}".format(m_precision))
        logging.info("mRecall: {:.6f}".format(m_recall))
        logging.info("mf1: {:.6f}".format(f1))

        self.ap = m_ap
        self.auc = m_auc
        self.dprime = dprime
        self.class_metrics = metrics

    def calculate_metrics(self,
                          y_predicted: np.array,
                          y_true: np.array) -> List[ClassMetrics]:
        classes_num = self.get_classes_num(y_true)

        metrics = []
        for class_idx in range(classes_num):
            class_stats = self.calculate_class_metrics(
                y_predicted,
                y_true,
                class_idx
            )
            metrics.append(class_stats)

        return metrics

    def get_classes_num(self, y: np.array):
        # 'y' has 2 dimensions (samples_num, classes_num)
        return y.shape[-1]

    def calculate_class_metrics(self,
                                y_predicted: np.array,
                                y_true: np.array,
                                class_index) -> ClassMetrics:
        # AP - Average precision
        ap = metrics.average_precision_score(
            y_true[:, class_index],
            y_predicted[:, class_index],
            average=None
        )

        # AUC
        if y_true.shape[1] > 1:
            auc = metrics.roc_auc_score(
                y_true[:, class_index],
                y_predicted[:, class_index],
                average=None
            )
        else:
            auc = 0

        # precission, recall and F1
        precision = metrics.precision_score(
            y_true[:, class_index],
            y_predicted[:, class_index] >= 0.5
        )
        recall = metrics.recall_score(
            y_true[:, class_index],
            y_predicted[:, class_index] >= 0.5
        )
        f1 = metrics.f1_score(
            y_true[:, class_index],
            y_predicted[:, class_index] >= 0.5
        )

        # confusion matrix
        tn, fp, fn, tp = metrics.confusion_matrix(
            y_true[:, class_index],
            y_predicted[:, class_index] >= 0.5
        ).ravel()

        return ClassMetrics(
            name=self.classes[class_index]['name'],
            ap=ap,
            auc=auc,
            precision=precision,
            recall=recall,
            f1=f1,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )

    def calculate_dprime(self, auc):
        standard_normal = norm()
        return standard_normal.ppf(auc) * np.sqrt(2.0)

    def log_classmetrics(self, metrics: List[ClassMetrics]):
        # sort metrics
        best_class_metrics = sorted(
            metrics, key=lambda classmetrics: -getattr(classmetrics, self.class_sort_key))
        worst_class_metrics = sorted(
            metrics, key=lambda classmetrics: getattr(classmetrics, self.class_sort_key))

        logging.info("  Best music classes:")
        for classmetrics in best_class_metrics[:self.show_top_classes]:
            logging.info(classmetrics)

        logging.info("  Worst music classes:")
        for classmetrics in worst_class_metrics[:self.show_top_classes]:
            logging.info(classmetrics)

    def save_classmetrics(self, metrics: List[ClassMetrics]):
        keys = metrics[0]._fields
        with open(self.classsmetrics_filepath, 'w') as output_file:
            writer = csv.DictWriter(output_file, keys)
            writer.writeheader()
            writer.writerows([m._asdict() for m in metrics])
