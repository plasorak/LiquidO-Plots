from rich import print as rprint
from dataclasses import dataclass
import numpy as np


@dataclass
class TimeCluster:
    start:float
    stop:float
    n_hits:int = 0
    max_hits:int = 0


    def grow(self, until, nhits):
        self.stop = until
        self.n_hits += nhits
        if nhits > self.max_hits:
            self.max_hits = nhits

@dataclass
class Cluster:
    t_min:float
    t_max:float
    x_min:float
    x_max:float
    y_min:float
    y_max:float
    n_hits:int
    hit_x:np.ndarray
    hit_y:np.ndarray
    hit_t:np.ndarray
    def __repr__(self):
        return f"Cluster(t_min={self.t_min}, t_max={self.t_max}, x_min={self.x_min}, x_max={self.x_max}, y_min={self.y_min}, y_max={self.y_max}, n_hits={self.n_hits})"

    @staticmethod
    def get_from_data(x_min, x_max, y_min, y_max, t_min, t_max, hit_x, hit_y, hit_t):
        mask =        (hit_t >= t_min) & (hit_t <= t_max)
        mask = mask & (hit_x >= x_min) & (hit_x <= x_max)
        mask = mask & (hit_y >= y_min) & (hit_y <= y_max)
        hit_t_ = hit_t[mask]
        hit_x_ = hit_x[mask]
        hit_y_ = hit_y[mask]

        t_min = np.min(hit_t_)
        t_max = np.max(hit_t_)
        x_min = np.min(hit_x_)
        x_max = np.max(hit_x_)
        y_min = np.min(hit_y_)
        y_max = np.max(hit_y_)

        n_hits = len(hit_x)

        return Cluster(
            t_min=t_min,
            t_max=t_max,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            n_hits=n_hits,
            hit_x=hit_x_,
            hit_y=hit_y_,
            hit_t=hit_t_,
        )

    @staticmethod
    def union(clusters):
        n_hits = 0
        t_min = clusters[0].t_min
        t_max = clusters[0].t_max
        x_min = clusters[0].x_min
        x_max = clusters[0].x_max
        y_min = clusters[0].y_min
        y_max = clusters[0].y_max

        for cluster in clusters:
            def biggest(a, b):
                if a is None or b is None:
                    return None
                return a if a > b else b
            def smallest(a, b):
                if a is None or b is None:
                    return None
                return a if a < b else b

            t_min = smallest(cluster.t_min, t_min)
            t_max = biggest (cluster.t_max, t_max)
            x_min = smallest(cluster.x_min, x_min)
            x_max = biggest (cluster.x_max, x_max)
            y_min = smallest(cluster.y_min, y_min)
            y_max = biggest (cluster.y_max, y_max)
            n_hits += cluster.n_hits

        return Cluster(
            t_min = t_min,
            t_max = t_max,
            x_min = x_min,
            x_max = x_max,
            y_min = y_min,
            y_max = y_max,
            n_hits = n_hits,
        )

class TimeClusterMaker:
    def __init__(self, time_binning, time_counts, hit_lifetime_ns):
        self.time_binning = time_binning
        self.time_counts = time_counts
        self.hit_lifetime_ns = hit_lifetime_ns
        self.clusters = []

    def run_dummy_clustering(self):
        current_cluster = None
        self.clusters = []
        from copy import deepcopy as dc
        import numpy as np

        for ibins, nhit in np.ndenumerate(self.time_counts):
            if current_cluster is not None: # already growing a cluster
                if nhit>0: # some hits, we continue to grow
                    current_cluster.grow(self.time_binning[ibins]+self.hit_lifetime_ns, nhit)
                elif nhit==0: # no hits, stop the cluster
                    self.clusters.append(dc(current_cluster))
                    current_cluster = None
            else: # not in a cluster
                if nhit>0: # need to start a new cluster
                    current_cluster = Cluster(self.time_binning[ibins])
                    current_cluster.grow(self.time_binning[ibins]+self.hit_lifetime_ns, nhit)
                elif nhit==0:
                    pass

    def run_edge_detector(self):
        current_cluster = None
        self.clusters = []
        from copy import deepcopy as dc
        import numpy as np
        diff = np.diff(self.time_counts)

        in_rising_edge = False

        for ibins, hdiff in np.ndenumerate(diff):
            nhit = self.time_counts[ibins]

            if hdiff>10: # above threshold, start a new cluster if not in rising edge already
                if in_rising_edge:
                    current_cluster.grow(self.time_binning[ibins]+self.hit_lifetime_ns, nhit)
                else:
                    if current_cluster is not None: # already growing a cluster, but not rising edge, save it, and start a new one
                        rprint(f"End of cluster at {self.time_binning[ibins]}")
                        self.clusters.append(dc(current_cluster))
                        current_cluster = None
                    rprint(f"New cluster at {self.time_binning[ibins]}")
                    current_cluster = TimeCluster(self.time_binning[ibins], self.time_binning[ibins])
                in_rising_edge = True

            else: # no need to start a cluster
                in_rising_edge = False

                if nhit>0: # some hits, we continue to grow
                    if current_cluster is not None:
                        current_cluster.grow(self.time_binning[ibins]+self.hit_lifetime_ns, nhit)
                    else:
                        rprint(f'Ignoring what looks like {nhit} noise hits at time {self.time_binning[ibins]}')
                elif current_cluster is not None: # done with the cluster
                    rprint(f"End of cluster at {self.time_binning[ibins]}")
                    self.clusters.append(dc(current_cluster))
                    current_cluster = None

        if current_cluster:
            self.clusters.append(dc(current_cluster))