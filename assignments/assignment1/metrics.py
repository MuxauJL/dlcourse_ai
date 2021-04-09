def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    total_count = len(ground_truth)
    true_positives_count = 0
    false_positives_count = 0
    false_negatives_count = 0
    correct_count = 0
    for i in range(total_count):
        gt = ground_truth[i]
        pred = prediction[i]
        if gt:
            if pred:
                true_positives_count += 1
                correct_count += 1
            else:
                false_negatives_count += 1
        else:
            if pred:
                false_positives_count += 1
            else:
                correct_count += 1

    precision = true_positives_count / (true_positives_count + false_positives_count)
    recall = true_positives_count / (true_positives_count + false_negatives_count)
    accuracy = correct_count / total_count
    f1 = 2 * precision * recall / (precision + recall)

    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    correct_count = 0
    total_count = len(ground_truth)
    for i in range(total_count):
        if ground_truth[i] == prediction[i]:
            correct_count += 1
    return correct_count / total_count
