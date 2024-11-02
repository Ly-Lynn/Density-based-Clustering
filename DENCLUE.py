import numpy as np
import math
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
    def __init__(self, x, y, h=0.3, xi=0.1, k=0.5, delta=0.1, max_times=100):
        self.x = x
        self.y = y
        self.n = len(x)
        assert(self.n == len(y))
        self.ps = [np.array([self.x[i], self.y[i]]) for i in range(self.n)]
        self.attrs = []
        self.bel = []
        self.is_out = []
        self.cluster_id = []
        
        # Lưu các tham số vào instance
        self.h = h  # Bandwidth parameter
        self.xi = xi  # Density threshold
        self.k = k  # Cluster connectivity threshold
        self.delta = delta  # Step size
        self.max_times = max_times

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
            mx = self.get_max(self.ps[i])
            self.attrs.append(mx)

    def next_pos(self, x):
        d = self.delta_f(x)
        if slen(d) == 0:
            return x
        return x + d * self.delta / slen(d)  # Sử dụng self.delta

    def get_max(self, start):
        x = start
        for i in range(self.max_times):  # Sử dụng self.max_times
            y = self.next_pos(x)
            if self.f_gauss(y) < self.f_gauss(x):
                break
            x = y
        return x

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
