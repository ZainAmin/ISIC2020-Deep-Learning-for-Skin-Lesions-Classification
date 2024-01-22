import matplotlib.pyplot as plt

def plot_roc(labels, preds, flag, model_name):
    """
    Plots Receiver Operating Characteristic (ROC) curves for a multi-class classification model.

    Parameters:
    - labels (list): True labels of the data.
    - preds (numpy.ndarray): Predicted probabilities for each class.
    - flag (bool): If True, display the ROC curves; otherwise, only calculate and return the Area Under the Curve (AUC).
    - model_name (str): Name of the model for plot title.

    Returns:
    - roc_auc (dict): A dictionary containing AUC values for each class.
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 3
    class_names = ['Melanoma', 'BCC', 'SCC']
    colors = ['#f06e93', '#327ba8', '#f0c96e']

    for i in range(n_classes):
        actuals = (np.array(labels) == i).astype(np.uint8)
        fpr[i], tpr[i], _ = roc_curve(actuals, preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    if flag:
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], color=colors[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(class_names[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic [ROC] for {0}'.format(model_name))
        plt.legend(loc="lower right")
        plt.show()

    return roc_auc
