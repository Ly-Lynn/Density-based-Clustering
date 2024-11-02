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

class Denclue2D(object):
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
        # Lưu các tham số vào instance
        self.h = h  # Bandwidth parameter
        self.xi = xi  # Density threshold
        self.k = k  # Cluster connectivity threshold
        self.delta = delta  # Step size
        self.max_times = max_times
        self.paths = []
    def f_gauss(self, x):
        s = 0
        for p in self.ps:
            s += k_gauss((x - p) / self.h)  # Sử dụng self.h thay vì H
        return s / (self.n * (self.h ** 2))

    def delta_f(self, x):
        s = np.array([0., 0.])
        for p in self.ps:
            s += k_gauss((x - p) / self.h) * (p - x)  # Sử dụng self.h
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
        return x + d * self.delta / slen(d)  # Sử dụng self.delta

    # Hill Climbing Algorithm
    def get_max(self, start):
        x = start
        positions = [x]  # Lưu các vị trí trong quá trình hội tụ
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
                if dist(self.attrs[i], self.attrs[j]) < 2 * self.delta:  # Sử dụng self.delta
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
            self.is_out.append(dens < self.xi)  # Sử dụng self.xi

    def merge_cluster(self):
        uf = UnionFind(len(self.attrs))
        is_higher = [self.f_gauss(p) >= self.xi for p in self.ps]  # Sử dụng self.xi
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.bel[i] != self.bel[j] \
                        and is_higher[i] \
                        and is_higher[j] \
                        and (not self.is_out[self.bel[i]]) \
                        and (not self.is_out[self.bel[j]]) \
                        and dist(self.ps[i], self.ps[j]) < self.k:  # Sử dụng self.k
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

    def plot_hill_climbing (self):
        plt.figure(figsize=(10, 6))
        
        plt.scatter([p[0] for p in self.ps], [p[1] for p in self.ps], c='lightgrey', s=30, label="Data points")
        
        for path in self.paths:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], 'o-', c='blue', alpha=0.5)
            
            for i in range(len(path) - 1):
                plt.arrow(path[i][0], path[i][1],
                          path[i + 1][0] - path[i][0],
                          path[i + 1][1] - path[i][1],
                          head_width=0.05, color='blue', alpha=0.5)

        local_max_positions = np.array(self.attrs)
        plt.scatter(local_max_positions[:, 0], local_max_positions[:, 1], c='red', s=100, label="Local Maxima")
        
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Hill Climbing Process for All Points")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_clusters(self, labels):
        
        points = np.array(self.ps)
        
        unique_labels = set(labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        
        plt.figure(figsize=(10, 6))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                plt.scatter(points[labels == label][:, 0], points[labels == label][:, 1], 
                            c='red', s=30, label="Noise", alpha=0.6)
            else:
                plt.scatter(points[labels == label][:, 0], points[labels == label][:, 1], 
                            color=color, s=30, label=f"Cluster {label}", alpha=0.6)
        
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Dataset after Clustering")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def __call__(self):
        _, labels = self.work()
        if self.plot:
            self.plot_hill_climbing()
            self.plot_clusters(labels)
        return self.get_result()