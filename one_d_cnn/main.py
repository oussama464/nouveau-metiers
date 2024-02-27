import pathlib
import numpy as np
from datapreper.dataprep import load_dataset
from models.cnn1d import evaluate_model

DATASET_PATH = pathlib.Path(__file__).parent.joinpath("UCI_HAR_Dataset")


def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print("Accuracy: %.3f%% (+/-%.3f)" % (m, s))


# run an experiment
def run_experiment(repeats=3):
    # load data
    trainX, trainy, testX, testy = load_dataset(DATASET_PATH)
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        print(">#%d: %.3f" % (r + 1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)


if __name__ == "__main__":
    run_experiment()
