import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS
import ipywidgets as widgets
from IPython.display import display, clear_output
from .visualization import Visualization

def generate_dataset(dataset_type, noise=0.1, n_samples=300):
    if dataset_type == "moons":
        X, _ = make_moons(n_samples=n_samples, noise=noise)  # Extract only X
        return X
    elif dataset_type == "circles":
        X, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.5)  # Extract only X
        return X
    elif dataset_type == "blobs":
        X, _ = make_blobs(n_samples=n_samples, centers=3, cluster_std=noise)  # Extract only X
        return X
    elif dataset_type == "anisotropic":
        X, _ = make_blobs(n_samples=n_samples, centers=3)  # Extract only X
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation)
        return X
    
class ClusteringComparison:
    def __init__(self, list_algs):
        self.dataset_types = ['moons', 'circles', 'blobs', 'anisotropic']
        # Initialize visualization object
        self.algs_list = list_algs
        self.visualization = Visualization(self.algs_list)
        # Widgets cho dataset
        self.dataset_widget = widgets.Dropdown(
            options=self.dataset_types,
            description='Dataset:',
            value='moons'
        )
        self.noise_widget = widgets.FloatSlider(
            value=0.1,
            min=0.01,
            max=0.5,
            step=0.01,
            description='Noise:'
        )
        for alg in list_algs:
            if alg == 'KMeans':
                self.kmeans_widget = widgets.IntSlider(
                    value=2,
                    min=2,
                    max=10,
                    description='KMeans clusters:'
                )
            if alg == 'Hierarchical':
                self.hierarchical_widget = widgets.IntSlider(
                    value=2,
                    min=2,
                    max=10,
                    description='Hierarchical clusters:'
                )
            if alg == 'DBSCAN':
                self.dbscan_eps_widget = widgets.FloatSlider(
                    value=0.3,
                    min=0.1,
                    max=1.0,
                    step=0.05,
                    description='DBSCAN eps:'
                )
                self.dbscan_min_samples_widget = widgets.IntSlider(
                    value=5,
                    min=2,
                    max=15,
                    description='DBSCAN min_samples:'
                )
            if alg == 'OPTICS':
                self.optics_min_samples_widget = widgets.IntSlider(
                    value=5,
                    min=2,
                    max=15,
                    description='OPTICS min_samples:'
                )
        
        
        # Layout

        self.dataset_controls = widgets.VBox([
            self.dataset_widget,
            self.noise_widget
        ])

        self.algorithm_controls = widgets.VBox([
            self.kmeans_widget if 'KMeans' in list_algs else widgets.VBox([]),
            self.hierarchical_widget if 'Hierarchical' in list_algs else widgets.VBox([]),
            self.dbscan_eps_widget if 'DBSCAN' in list_algs else widgets.VBox([]),
            self.dbscan_min_samples_widget if 'DBSCAN' in list_algs else widgets.VBox([]),
            self.optics_min_samples_widget if 'OPTICS' in list_algs else widgets.VBox([])
        ])
        
        # Update button
        self.update_button = widgets.Button(description='Update Plots')
        self.update_button.on_click(self.update_plots)
        
        # Display widgets
        display(widgets.VBox([
            self.dataset_controls,
            self.algorithm_controls,
            self.update_button
        ]))
        self.X = generate_dataset(
            self.dataset_widget.value,
            self.noise_widget.value
        )
        self.X = np.array(self.X, dtype=np.float64)

    
    def update_plots(self, _):
        clear_output(wait=True)
        
        # Display widgets again
        display(widgets.VBox([
            self.dataset_controls,
            self.algorithm_controls,
            self.update_button
        ]))
        
        titles = []
        titles.append(f'Original Data\nDataset: {self.dataset_widget.value}, Noise: {self.noise_widget.value:.2f}')
        labels_list = []
        labels_list.append(np.zeros(len(self.X)))
        # Perform clustering
        if self.kmeans_widget.value > 0:
            kmeans = KMeans(n_clusters=self.kmeans_widget.value)
            kmeans_labels = kmeans.fit_predict(self.X)
            titles.append(f'KMeans\nClusters: {self.kmeans_widget.value}')
            labels_list.append(kmeans_labels)
        if self.hierarchical_widget.value > 0:
            hierarchical = AgglomerativeClustering(n_clusters=self.hierarchical_widget.value)
            hierarchical_labels = hierarchical.fit_predict(self.X)
            titles.append(f'Hierarchical\nClusters: {self.hierarchical_widget.value}')
            labels_list.append(hierarchical_labels)
        if self.dbscan_eps_widget.value > 0:
            dbscan = DBSCAN(eps=self.dbscan_eps_widget.value, min_samples=self.dbscan_min_samples_widget.value)
            dbscan_labels = dbscan.fit_predict(self.X)
            titles.append(f'DBSCAN\neps: {self.dbscan_eps_widget.value}, min_samples: {self.dbscan_min_samples_widget.value}')
            labels_list.append(dbscan_labels)
        if self.optics_min_samples_widget.value > 0:
            optics = OPTICS(min_samples=self.optics_min_samples_widget.value)
            optics_labels = optics.fit_predict(self.X)
            titles.append(f'OPTICS\nmin_samples: {self.optics_min_samples_widget.value}')
            labels_list.append(optics_labels)
        
        # Call visualize method
        self.visualization.visualize(self.X, self.algs_list, labels_list, titles)

if __name__ == '__main__':
    list_algs = ['KMeans', 'Hierarchical', 'DBSCAN', 'OPTICS']
    comp = ClusteringComparison(list_algs)