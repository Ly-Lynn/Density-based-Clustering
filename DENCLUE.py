import numpy as np
import math
import matplotlib.pyplot as plt

def sqrs(x):
    return (x ** 2).sum()

def slen(x):
    return sqrs(x) ** 0.5

def dist(x, y):
    return sqrs(x - y) ** 0.5

def k_gauss(x):
    return math.exp(-0.5 * sqrs(x)) / (2 * math.pi)

def get_z(x, y, f):
    m, n = x.shape
    z = np.empty((m, n))
    for i in range(m):
        for j in range(n):
            z[i, j] = f(x[i, j], y[i, j])
    return z

class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.sid = list(range(n))
        self.pos = list(range(n))
        
    def find(self, x):
        if self.p[x] == x:
            return x
        self.p[x] = self.find(self.p[x])
        return self.p[x]
    
    def merge(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px != py:
            self.p[px] = py
            
    def arrange(self):
        n = len(self.p)
        for i in range(n):
            self.sid[i] = self.find(i)
        cnt = 0
        mp = {}
        self.pos = []
        for i in range(n):
            if self.sid[i] == i:
                mp[i] = cnt
                self.pos.append(i)
                cnt += 1
        for i in range(n):
            self.sid[i] = mp[self.sid[i]]

class Denclue2D:
    def __init__(self, x, y, h=0.3, xi=0.1, k=0.5, delta=0.1, max_times=100, plot=False):
        self.x = x
        self.y = y
        self.n = len(x)
        assert(self.n == len(y))
        self.ps = [np.array([self.x[i], self.y[i]]) for i in range(self.n)]
        self.attrs = []
        self.bel = []
        self.is_out = []
        self.cluster_id = []
        self.plot = plot
        self.h = h
        self.xi = xi
        self.k = k
        self.delta = delta
        self.max_times = max_times
        self.paths = []

    def f_gauss(self, x):
        s = 0
        for p in self.ps:
            s += k_gauss((x - p) / self.h)
        return s / (self.n * (self.h ** 2))

    def delta_f(self, x):
        s = np.array([0., 0.])
        for p in self.ps:
            s += k_gauss((x - p) / self.h) * (p - x)
        return s / ((self.h ** 4) * self.n)
    
    def climbs(self):
        for i in range(self.n):
            max_position, path = self.get_max(self.ps[i])
            self.attrs.append(max_position)
            self.paths.append(path)

    def next_pos(self, x):
        d = self.delta_f(x)
        if slen(d) == 0:
            return x
        return x + d * self.delta / slen(d)

    def get_max(self, start):
        x = start
        positions = [x]
        for i in range(self.max_times):
            y = self.next_pos(x)
            positions.append(y)
            if self.f_gauss(y) < self.f_gauss(x):
                break
            x = y
        return x, positions

    def merge_same(self):
        uf = UnionFind(self.n)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if dist(self.attrs[i], self.attrs[j]) < 2 * self.delta:
                    uf.merge(i, j)
        uf.arrange()
        new_attrs = []
        for position in uf.pos:
            new_attrs.append(self.attrs[position])
        self.attrs = new_attrs
        for i in range(self.n):
            self.bel.append(uf.sid[i])

    def tag_outs(self):
        for at in self.attrs:
            dens = self.f_gauss(at)
            self.is_out.append(dens < self.xi)

    def merge_cluster(self):
        uf = UnionFind(len(self.attrs))
        is_higher = [self.f_gauss(p) >= self.xi for p in self.ps]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.bel[i] != self.bel[j] \
                        and is_higher[i] \
                        and is_higher[j] \
                        and (not self.is_out[self.bel[i]]) \
                        and (not self.is_out[self.bel[j]]) \
                        and dist(self.ps[i], self.ps[j]) < self.k:
                    uf.merge(self.bel[i], self.bel[j])
        uf.arrange()
        for i in range(len(self.attrs)):
            self.cluster_id.append(uf.sid[i])

    def get_result(self):
        res = []
        for i in range(self.n):
            if self.is_out[self.bel[i]]:
                res.append(-1)
            else:
                res.append(self.cluster_id[self.bel[i]])
        no = [-1 for i in range(len(self.attrs))]
        cnt = 0
        for i in range(len(res)):
            if res[i] != -1:
                if no[res[i]] == -1:
                    no[res[i]] = cnt
                    cnt += 1
                res[i] = no[res[i]]
        return cnt, res
    
    def work(self):
        self.climbs()
        self.merge_same()
        self.tag_outs()
        self.merge_cluster()
        return self.get_result()

    def plot_hill_climbing(self, ax):
        ax.scatter([p[0] for p in self.ps], [p[1] for p in self.ps], c='lightgrey', s=30, label="Data points")
        
        for path in self.paths:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], 'o-', c='blue', alpha=0.5)
            for i in range(len(path) - 1):
                ax.arrow(path[i][0], path[i][1],
                         path[i + 1][0] - path[i][0],
                         path[i + 1][1] - path[i][1],
                         head_width=0.05, color='blue', alpha=0.5)

        local_max_positions = np.array(self.attrs)
        ax.scatter(local_max_positions[:, 0], local_max_positions[:, 1], c='red', s=100, label="Local Maxima")
        
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title("Hill Climbing Process")
        ax.legend()
        ax.grid(True)

    def plot_clusters(self, ax, labels):
        points = np.array(self.ps)
        unique_labels = set(labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                ax.scatter(points[np.array(labels) == label][:, 0], points[np.array(labels) == label][:, 1], 
                           c='red', s=30, label="Noise", alpha=0.6)
            else:
                ax.scatter(points[np.array(labels) == label][:, 0], points[np.array(labels) == label][:, 1], 
                           color=color, s=30, label=f"Cluster {label}", alpha=0.6)
        
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title("Dataset after Clustering")
        ax.legend()
        ax.grid(True)
    
    def __call__(self):
        _, labels = self.work()
        if self.plot:
            fig, axes = plt.subplots(1, 1, figsize=(10, 10))
            # self.plot_hill_climbing(axes[0])
            self.plot_clusters(axes, labels)
            plt.show()
        return self.get_result()
