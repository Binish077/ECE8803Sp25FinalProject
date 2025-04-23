import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def analyze_predictions(predictions_file, labels_file, output_csv, metrics_file):
    """
    Analyze predictions, save them to a CSV file, and calculate accuracy and F1 score.

    Inputs:
        predictions_file (str): Path to the .npy file containing predictions.
        labels_file (str): Path to the .npy file containing true labels.
        output_csv (str): Path to save the predictions and labels as a CSV file.
        metrics_file (str): Path to save the accuracy and F1 score.

    Note: If you don't want to save the stats in a file, evaluate_model function in train.py can be used instead.
    """

    # load pred and labels
    predictions = np.load(predictions_file)
    labels = np.load(labels_file)

    # save predictions
    data = {
        'Predictions': [list(pred) for pred in predictions],
        #'Labels': labels
    }
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Predictions and labels saved to {output_csv}")

    # get statistics of model performance
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')

    # save statistics
    with open(metrics_file, 'w') as f:
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"F1 Score: {f1:.2f}\n")
    print(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    # testing:
    # analyze_predictions(
    #     predictions_file='./save/test_predictions.npy',
    #     labels_file='./save/test_labels.npy',
    #     output_csv='predictions_analysis.csv',
    #     metrics_file='metrics.txt'
    # )