import numpy as np
import matplotlib.pyplot as plt
import csv


class Optics:
    def __init__(self, minSamples=5, maxEps=np.inf):
        self.minSamples = minSamples  # Minimum points for a core point
        self.maxEps = maxEps  # Max distance to consider neighbor
        self.labels_ = None  # Cluster labels for each point
        self.reachability_ = None  # Reachability distances
        self.ordering_ = None  # Order points were processed
        self.coreDistances_ = None  # Core distances for each point

    def _distance(self, p1, p2):
        distSquared = 0
        for i in range(len(p1)):
            distSquared += (p1[i] - p2[i]) ** 2
        return distSquared**0.5  # Euclidean distance

    def _neighbors(self, X, pointIdx):
        neighbors = [
            (i, self._distance(X[pointIdx], X[i]))
            for i in range(len(X))
            if i != pointIdx and self._distance(X[pointIdx], X[i]) <= self.maxEps
        ]
        neighbors.sort(key=lambda x: x[1])
        return neighbors  # Sorted neighbors by distance

    def _coreDistance(self, neighbors):
        if len(neighbors) < self.minSamples - 1:
            return None  # Not enough neighbors to be core
        return neighbors[self.minSamples - 2][1]  # Distance to minSamples-th neighbor

    def _reachabilityDistance(self, coreDist, neighborDist):
        if coreDist is None:
            return None
        return max(coreDist, neighborDist)  # Reachability distance formula

    def fit(self, X):
        nPoints = len(X)
        processed = np.zeros(nPoints, dtype=bool)  # Track processed points
        reachability = np.full(nPoints, np.inf)
        coreDistances = np.full(nPoints, np.inf)
        ordering = []
        seeds = []

        def insertSorted(seeds, pointIdx, reachDist):
            # Insert point in seeds list sorted by reachability
            for i, (d, _) in enumerate(seeds):
                if reachDist < d:
                    seeds.insert(i, (reachDist, pointIdx))
                    return
            seeds.append((reachDist, pointIdx))

        for pointIdx in range(nPoints):
            if processed[pointIdx]:
                continue

            neighbors = self._neighbors(X, pointIdx)
            processed[pointIdx] = True
            coreDist = self._coreDistance(neighbors)
            coreDistances[pointIdx] = coreDist if coreDist is not None else np.inf
            ordering.append(pointIdx)

            if coreDist is not None:
                for neighborIdx, neighborDist in neighbors:
                    if not processed[neighborIdx]:
                        newReach = self._reachabilityDistance(coreDist, neighborDist)
                        if newReach < reachability[neighborIdx]:
                            reachability[neighborIdx] = newReach
                            insertSorted(seeds, neighborIdx, newReach)

                while seeds:
                    _, currentIdx = seeds.pop(0)
                    if processed[currentIdx]:
                        continue

                    neighbors = self._neighbors(X, currentIdx)
                    processed[currentIdx] = True
                    coreDist = self._coreDistance(neighbors)
                    coreDistances[currentIdx] = (
                        coreDist if coreDist is not None else np.inf
                    )
                    ordering.append(currentIdx)

                    if coreDist is not None:
                        for neighborIdx, neighborDist in neighbors:
                            if not processed[neighborIdx]:
                                newReach = self._reachabilityDistance(
                                    coreDist, neighborDist
                                )
                                if newReach < reachability[neighborIdx]:
                                    reachability[neighborIdx] = newReach
                                    insertSorted(seeds, neighborIdx, newReach)

        self.ordering_ = np.array(ordering)  # Save processing order
        self.reachability_ = reachability
        self.coreDistances_ = coreDistances
        self.labels_ = self._extractClusters(X)  # Extract clusters from reachability

        return self

    def _extractClusters(self, X, minClusterSize=0.05):
        nPoints = len(X)
        minPoints = max(2, int(minClusterSize * nPoints))
        labels = np.full(nPoints, -1)  # Initialize as noise
        orderedReach = self.reachability_[self.ordering_]
        clusterId = 0
        clusterPoints = []

        threshold = np.percentile(
            orderedReach[orderedReach != np.inf], 75
        )  # Simple valley threshold

        for i, idx in enumerate(self.ordering_):
            if orderedReach[i] <= threshold:
                clusterPoints.append(idx)
            else:
                if len(clusterPoints) >= minPoints:
                    for p in clusterPoints:
                        labels[p] = clusterId
                    clusterId += 1
                clusterPoints = []

        if len(clusterPoints) >= minPoints:
            for p in clusterPoints:
                labels[p] = clusterId

        return labels


def loadCsv(filePath):
    data = []
    with open(filePath, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row:
                data.append([float(row[0]), float(row[1])])
    return np.array(data)


if __name__ == "__main__":
    data = loadCsv("datagalaxy.csv")
    optics = Optics(minSamples=5)
    optics.fit(data)
    labels = optics.labels_

    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="rainbow", edgecolor="k")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("OPTICS Clustering")
    plt.show()

    print("Number of clusters found:", len(np.unique(labels[labels != -1])))
    print("Number of noise points:", np.sum(labels == -1))
