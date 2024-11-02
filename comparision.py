import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS
import ipywidgets as widgets
from IPython.display import display, clear_output
from visualization import Visualization

def generate_dataset(dataset_type, noise=0.1, n_samples=300):
    if dataset_type == "moons":
        return make_moons(n_samples=n_samples, noise=noise)
    elif dataset_type == "circles":
        return make_circles(n_samples=n_samples, noise=noise, factor=0.5)
    elif dataset_type == "blobs":
        return make_blobs(n_samples=n_samples, centers=3, cluster_std=noise)
    elif dataset_type == "anisotropic":
        X, y = make_blobs(n_samples=n_samples, centers=3)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation)
        return X, y
    
class ClusteringComparison:
    def __init__(self):
        self.dataset_types = ['moons', 'circles', 'blobs', 'anisotropic']
        
        # Initialize visualization object
        self.visualization = Visualization()
        
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
        
        # Widgets cho KMeans
        self.kmeans_n_clusters = widgets.IntSlider(
            value=2,
            min=2,
            max=10,
            description='K-Means clusters:'
        )
        
        # Widgets cho Hierarchical
        self.hierarchical_n_clusters = widgets.IntSlider(
            value=2,
            min=2,
            max=10,
            description='Hierarchical clusters:'
        )
        
        # Widgets cho DBSCAN
        self.dbscan_eps = widgets.FloatSlider(
            value=0.3,
            min=0.1,
            max=1.0,
            step=0.05,
            description='DBSCAN eps:'
        )
        self.dbscan_min_samples = widgets.IntSlider(
            value=5,
            min=2,
            max=15,
            description='DBSCAN min_samples:'
        )
        
        # Widgets cho OPTICS
        self.optics_min_samples = widgets.IntSlider(
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
            self.kmeans_n_clusters,
            self.hierarchical_n_clusters,
            self.dbscan_eps,
            self.dbscan_min_samples,
            self.optics_min_samples
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
        
    def update_plots(self, _):
        clear_output(wait=True)
        
        # Display widgets again
        display(widgets.VBox([
            self.dataset_controls,
            self.algorithm_controls,
            self.update_button
        ]))
        
        # Generate dataset
        X, _ = generate_dataset(
            self.dataset_widget.value,
            self.noise_widget.value
        )
        
        # Perform clustering
        kmeans = KMeans(n_clusters=self.kmeans_n_clusters.value)
        hierarchical = AgglomerativeClustering(n_clusters=self.hierarchical_n_clusters.value)
        dbscan = DBSCAN(eps=self.dbscan_eps.value, min_samples=self.dbscan_min_samples.value)
        optics = OPTICS(min_samples=self.optics_min_samples.value)
        
        # Fit and get labels
        kmeans_labels = kmeans.fit_predict(X)
        hierarchical_labels = hierarchical.fit_predict(X)
        dbscan_labels = dbscan.fit_predict(X)
        optics_labels = optics.fit_predict(X)
        
        # Titles and labels list
        titles = [
            f'Original Data\nDataset: {self.dataset_widget.value}, Noise: {self.noise_widget.value:.2f}',
            f'K-Means\nClusters: {self.kmeans_n_clusters.value}',
            f'Hierarchical\nClusters: {self.hierarchical_n_clusters.value}',
            f'DBSCAN\neps: {self.dbscan_eps.value}, min_samples: {self.dbscan_min_samples.value}',
            f'OPTICS\nmin_samples: {self.optics_min_samples.value}'
        ]
        
        labels_list = [
            np.zeros(len(X)),
            kmeans_labels,
            hierarchical_labels,
            dbscan_labels,
            optics_labels
        ]
        
        # Call visualize method
        self.visualization.visualize(X, labels_list, titles)
