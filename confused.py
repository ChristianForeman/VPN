import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sklearn.metrics
from sklearn.utils.multiclass import unique_labels

def read_classes_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    classes = [line.strip().split('\t')[1] for line in lines]
    return classes

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
    else:
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(len(classes)+2, len(classes)))
    sns.set(font_scale=1.2)
    cm *= 100
    sns.heatmap(cm, annot=True, cmap=cmap, fmt=".1f", linewidths=.5, square=True,
                xticklabels=classes, yticklabels=classes, cbar_kws={"shrink": 0.75})
    plt.title(title, fontweight='bold', fontsize='27')
    plt.ylabel('True label', fontweight='bold', fontsize='24')
    plt.xlabel('Predicted label', fontweight='bold', fontsize='24')
    plt.show()

def read_outputtinged(file_path):
    # Read from the txt file
    ground_truth= []
    prediction = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse each line and populate the lists
    for line in lines:
        values = line.strip().split(',')
        ground_truth.append(int(values[0].strip()))
        prediction.append(int(values[1].strip()))
    
    return ground_truth, prediction


y_true, y_pred = read_outputtinged('outputinged.txt')
print("F1-Score:", sklearn.metrics.f1_score(y_true, y_pred, average='macro'))
print("Accuracy:", sklearn.metrics.accuracy_score(y_true, y_pred))
plot_confusion_matrix(y_true, y_pred, read_classes_from_file("key.txt"), True)