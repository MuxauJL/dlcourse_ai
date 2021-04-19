def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    correct_count = 0
    total_count = len(ground_truth)
    for i in range(total_count):
        if ground_truth[i] == prediction[i]:
            correct_count += 1
    return correct_count / total_count
