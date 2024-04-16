import numpy as np
from rich import print as rprint
from dataclasses import dataclass

@dataclass
class SpaceCluster:
    neighbours:np.ndarray
    n_hits = 0
    hit_x:np.ndarray
    hit_y:np.ndarray

    def __post_init__(self):
        self.n_hits = len(self.neighbours)

    def grow(self, new_neighbours):
        self.neighbours += new_neighbours
        self.n_hits += len(new_neighbours)

    def get_min_max(self):
        x_min = np.min(self.hit_x[self.neighbours])
        x_max = np.max(self.hit_x[self.neighbours])
        y_min = np.min(self.hit_y[self.neighbours])
        y_max = np.max(self.hit_y[self.neighbours])
        return x_min, x_max, y_min, y_max

class DBScan:
    def __init__(self, eps, min_hits, hit_x, hit_y):
        self.eps = eps
        self.min_hits = min_hits
        self.hit_x = hit_x
        self.hit_y = hit_y
        self.n_hits = hit_x.shape[0]
        if hit_y.shape[0] != self.n_hits:
            raise ValueError("hit_x and hit_y must have the same number of hits")
        rprint(f"DBScan: {self.n_hits} hits")
        self.clusters = []

    def distance(self, i, j):
        hit_x_i = self.hit_x[i]
        hit_x_j = self.hit_x[j]

        hit_y_i = self.hit_y[i]
        hit_y_j = self.hit_y[j]

        return np.sqrt(
           (hit_x_i - hit_x_j)**2 +
           (hit_y_i - hit_y_j)**2
        )

    def region_query(self, i):
        neighbours = []
        for j in range(self.n_hits):
            if self.distance(i, j) < self.eps:
                neighbours += [j]
        return np.array(neighbours)

    def run(self):
        label = np.zeros(self.n_hits)
        # -1 -> noise
        # 0 -> unseen
        # 1,2,3,4,... -> cluster_id
        cluster_id = 1

        for i in range(self.n_hits):
            if label[i] != 0:
                continue

            label[i] = 0

            neighbours = self.region_query(i)
            if neighbours.shape[0] < self.min_hits-1:
                label[i] = -1
                continue

            rprint(f"New cluster {cluster_id} from hit {i} with {len(neighbours)} hits, initially")
            cluster_id += 1
            label[i] = cluster_id

            while neighbours.shape[0] > 0:
                neighbour = neighbours[-1]
                neighbours = neighbours[:-1]
                if label[neighbour] == -1:
                    # not sure how this can ever happen?
                    rprint("This should not happen")
                    label[neighbour] = cluster_id

                if label[neighbour] != 0:
                    # already assigned to the cluster
                    continue

                label[neighbour] = cluster_id
                new_neighbours = self.region_query(neighbour)
                if new_neighbours.shape[0] >= self.min_hits:
                    a = np.concatenate((neighbours, new_neighbours))
                    neighbours = np.unique(a)
                    chuck = []
                    for i, hit in np.ndenumerate(neighbours):
                        if label[hit] != 0:
                            chuck += [i]
                    neighbours = np.delete(neighbours, chuck)


        for the_label in np.unique(label):
            if the_label == -1:
                continue
            if the_label == 0:
                rprint("Forgotten hit!!")

            indices = []
            for index, x in np.ndenumerate(label):
                if x == the_label:
                    indices.append(index)
            cluster = SpaceCluster(
                neighbours = np.array(indices),
                hit_x = self.hit_x,
                hit_y = self.hit_y
            )
            self.clusters.append(cluster)
