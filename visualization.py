import numpy as np
import matplotlib.pyplot as plt


class Visualization:
    def __init__(self, algs_list, figsize=(20, 12)):
        self.figsize = figsize
        self.algs_list = algs_list
        self.fig = None
        self.axes = None

    def plot_clusters(self, X, labels, title, ax):
        unique_labels = np.unique(labels)
        
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
        
        if len(unique_labels) > 1:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        ax.axis('equal')

    def plot_reachability(self, reachability, labels, ax):
        colors = ['red' if label == -1 else plt.cm.viridis(label / (np.max(labels) + 1))
                  for label in labels]
        ax.bar(range(len(reachability)), reachability, color=colors, edgecolor='black')
        ax.set_title("OPTICS Reachability Plot")
        ax.set_xlabel("Points ordered by OPTICS")
        ax.set_ylabel("Reachability Distance")

    def visualize(self, X, labels_list, titles, reachability=None, optics_labels=None):
        if self.fig is not None:
            plt.close(self.fig)

        lens = len(self.algs_list)
        rows = lens // 3 + 1 if lens % 3 != 0 else lens // 3
        cols = 3 if lens >= 3 else lens

        self.fig, self.axes = plt.subplots(rows, cols, figsize=self.figsize)
        self.axes = self.axes.ravel()
        
        for ax, labels, title in zip(self.axes[:-1], labels_list, titles):
            self.plot_clusters(X, labels, title, ax)
        
        # If OPTICS is used, plot the reachability plot on the last subplot
        if 'OPTICS' in self.algs_list and reachability is not None:
            self.plot_reachability(reachability, optics_labels, self.axes[-1])
        else:
            # Hide the last empty plot area if not using OPTICS
            self.axes[-1].axis('off')
        
        plt.tight_layout()
        plt.show()
