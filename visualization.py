import numpy as np
import matplotlib.pyplot as plt


class Visualization:
    def __init__(self, type, figsize=(20, 12)):
        self.figsize = figsize
        self.type = type
        self.fig = None
        self.axes = None

    def plot_clusters(self, X, labels, title, ax):
        unique_labels = np.unique(labels)
        
        # Define colors
        if len(unique_labels) == 1:
            colors = ['red' if unique_labels[0] == -1 else 'blue']
        else:
            colors = ['red' if label == -1 else plt.cm.viridis(i / (len(unique_labels) - 1)) 
                      for i, label in enumerate(unique_labels)]
        
        for i, label in enumerate(unique_labels):
            color = colors[i]
            if label == -1:
                ax.scatter(X[labels == label, 0], X[labels == label, 1], c='red', s=30, label='Noise')
            else:
                ax.scatter(X[labels == label, 0], X[labels == label, 1], color=color, s=30, label=f'Cluster {label}')
        
        ax.set_title(title)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        
        # Only show legend if there are more than one unique label
        if len(unique_labels) > 1:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        ax.axis('equal')

    def visualize(self, X, labels_list, titles):
        # Close previous plot
        if self.fig is not None:
            plt.close(self.fig)
        
        # Create new figure
        self.fig, self.axes = plt.subplots(2, 3, figsize=self.figsize)
        self.axes = self.axes.ravel()
        
        for ax, labels, title in zip(self.axes[:-1], labels_list, titles):
            self.plot_clusters(X, labels, title, ax)
        
        # Hide the last empty plot area
        self.axes[-1].axis('off')
        
        plt.tight_layout()
        plt.show()