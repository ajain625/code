import numpy as np
import torch


# Metrics
# Statistical Parity Difference = P(y_hat = 1 | A = 1) - P(y_hat = 1 | A = 0) = proportion of positives on hard task - proportion of positives on easy task
# Equal Opportunity Difference = P(y_hat = 1 | A = 1, Y = 1) - P(y_hat = 1 | A = 0, Y = 1) = proportion of positives on hard task for positive examples - proportion of positives on easy task for positive examples
# Equal Accuracy Difference = Accuracy Easy - Accuracy Hard
# Equalized Odds Difference = max(TPR difference, TNR difference)
# Predicted Parity = Precision (Easy) - Precision (Hard), where Precision = TP / (TP + FP)

def get_conf_matrix(outputs, labels):
    # outputs: (batch_size, num_classes)
    # labels: (batch_size)
    # returns: (num_classes, num_classes) [[TP, FN], [FP, TN]
    preds = torch.argmax(outputs, dim=1)
    conf_matrix = torch.zeros((outputs.shape[1], outputs.shape[1]))
    for i in reversed(range(outputs.shape[1])):
        for j in reversed(range(outputs.shape[1])):
            conf_matrix[i, j] = torch.sum((labels == i) * (preds == j))
    return conf_matrix.numpy()

def statistical_parity_difference(conf1, conf2):
    # conf1: (num_classes, num_classes) [[TP, FN], [FP, TN]], easy task
    # conf2: (num_classes, num_classes) [[TP, FN], [FP, TN]], hard task
    return (conf1[0, 0] + conf1[1, 0])/ np.sum(conf1) - (conf2[0, 0] + conf2[1, 0])/ np.sum(conf2)

def equal_opportunity_difference(conf1, conf2):
    # conf1: (num_classes, num_classes) [[TP, FN], [FP, TN]], easy task
    # conf2: (num_classes, num_classes) [[TP, FN], [FP, TN]], hard task
    return conf1[0, 0]/ (conf1[0, 0] + conf1[0, 1]) - conf2[0, 0]/ (conf2[0, 0] + conf2[0, 1])

def equal_accuracy_difference(conf1, conf2):
    # conf1: (num_classes, num_classes) [[TP, FN], [FP, TN]], easy task
    # conf2: (num_classes, num_classes) [[TP, FN], [FP, TN]], hard task
    return (conf1[0, 0] + conf1[1, 1])/ np.sum(conf1) - (conf2[0, 0] + conf2[1, 1])/ np.sum(conf2)

def equalized_odds_difference(conf1, conf2):
    # conf1: (num_classes, num_classes) [[TP, FN], [FP, TN]], easy task
    # conf2: (num_classes, num_classes) [[TP, FN], [FP, TN]], hard task
    TPR_diff = conf1[0, 0]/ (conf1[0, 0] + conf1[0, 1]) - conf2[0, 0]/ (conf2[0, 0] + conf2[0, 1])
    TNR_diff = conf1[1, 1]/ (conf1[1, 0] + conf1[1, 1]) - conf2[1, 1]/ (conf2[1, 0] + conf2[1, 1])
    return max(abs(TPR_diff), abs(TNR_diff))

def predicted_parity(conf1, conf2):
    # conf1: (num_classes, num_classes) [[TP, FN], [FP, TN]], easy task
    # conf2: (num_classes, num_classes) [[TP, FN], [FP, TN]], hard task
    precision_easy = conf1[0, 0]/ (conf1[0, 0] + conf1[1, 0])
    precision_hard = conf2[0, 0]/ (conf2[0, 0] + conf2[1, 0])
    return precision_easy - precision_hard