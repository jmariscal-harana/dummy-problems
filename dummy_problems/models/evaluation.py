import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(conf_mat):
    """Create confusion matrix plot."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_mat.cpu().numpy(), annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    return fig

def plot_roc_curves(fpr, tpr, num_classes):
    """Create ROC curves plot."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(num_classes):
        ax.plot(fpr[i].cpu(), tpr[i].cpu(), label=f'Class {i}')
    
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()

    return fig