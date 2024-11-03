import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS
import ipywidgets as widgets
from IPython.display import display, clear_output
from .visualization import Visualization

def generate_dataset(dataset_type, noise=0.1, n_samples=100):
    if dataset_type == "moons":
        X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=42)  
        return X
    elif dataset_type == "circles":
        X, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
        return X
    elif dataset_type == "blobs":
        X, _ = make_blobs(n_samples=n_samples, centers=3, cluster_std=noise, random_state=42)
        return X
    elif dataset_type == "anisotropic":
        X, _ = make_blobs(n_samples=n_samples, centers=3, random_state=42)  
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation)
        return X
    
class ClusteringComparison:
    def __init__(self, list_algs, custom_dataset=None):
        self.custom_dataset = custom_dataset
        self.dataset_types = ['moons', 'circles', 'blobs', 'anisotropic']
        self.algs_list = list_algs
        self.visualization = Visualization(self.algs_list)
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
                    value=0.25,
                    min=0.1,
                    max=1.0,
                    step=0.05,
                    description='DBSCAN eps:'
                )
                self.dbscan_min_samples_widget = widgets.IntSlider(
                    value=4,
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
        
        self.update_button = widgets.Button(description='Update Plots')
        self.update_button.on_click(self.update_plots)
        
        display(widgets.VBox([
            self.dataset_controls,
            self.algorithm_controls,
            self.update_button
        ]))
        self.X = None

    def update_plots(self, _):
        clear_output(wait=True)
        display(widgets.VBox([
            self.dataset_controls,
            self.algorithm_controls,
            self.update_button
        ]))
        if self.custom_dataset is not None:
            self.X = self.custom_dataset
        else:
            self.X = generate_dataset(self.dataset_widget.value, self.noise_widget.value)
            self.X = np.array(self.X, dtype=np.float64)
        
        titles = [f'Original Data\nDataset: {self.dataset_widget.value}, Noise: {self.noise_widget.value:.2f}']
        labels_list = [np.zeros(len(self.X))]

        reachability = None
        optics_labels = None

        if 'KMeans' in self.algs_list:
            kmeans = KMeans(n_clusters=self.kmeans_widget.value)
            labels_list.append(kmeans.fit_predict(self.X))
            titles.append(f'KMeans\nClusters: {self.kmeans_widget.value}')

        if 'Hierarchical' in self.algs_list:
            hierarchical = AgglomerativeClustering(n_clusters=self.hierarchical_widget.value)
            labels_list.append(hierarchical.fit_predict(self.X))
            titles.append(f'Hierarchical\nClusters: {self.hierarchical_widget.value}')

        if 'DBSCAN' in self.algs_list:
            dbscan = DBSCAN(eps=self.dbscan_eps_widget.value, min_samples=self.dbscan_min_samples_widget.value)
            labels_list.append(dbscan.fit_predict(self.X))
            titles.append(f'DBSCAN\neps: {self.dbscan_eps_widget.value}, min_samples: {self.dbscan_min_samples_widget.value}')
        
        if 'OPTICS' in self.algs_list:
            optics = OPTICS(min_samples=self.optics_min_samples_widget.value)
            optics_labels = optics.fit_predict(self.X)
            reachability = optics.reachability_[optics.ordering_]
            labels_list.append(optics_labels)
            titles.append(f'OPTICS\nmin_samples: {self.optics_min_samples_widget.value}')

        self.visualization.visualize(self.X, labels_list, titles, reachability, optics_labels)


if __name__ == '__main__':
    list_algs = ['KMeans', 'Hierarchical', 'DBSCAN', 'OPTICS']
    comp = ClusteringComparison(list_algs)