def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    correct_count = 0
    total_count = len(ground_truth)
    for i in range(total_count):
        if ground_truth[i] == prediction[i]:
            correct_count += 1
    return correct_count / total_count
