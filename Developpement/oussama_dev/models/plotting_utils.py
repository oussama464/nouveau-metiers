import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt


def plot_model_confusion_matrix(y_test, predictions, score):
    cm = metrics.confusion_matrix(y_test, predictions, labels=[True, False])
    plt.figure(figsize=(9, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        linewidths=0.5,
        square=True,
        cmap="Blues_r",
        xticklabels=["True", "False"],
        yticklabels=["True", "False"],
    )
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    all_sample_title = f"Accuracy Score: {score}"
    plt.title(all_sample_title, size=15)
    plt.show()
