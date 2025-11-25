import numpy as np
import matplotlib.pyplot as plt
import csv
import heapq


# Load csv file with at least 2 columns (x and y values)
def loadCsvFile(filePath):
    dataPoints = []
    with open(filePath, "r") as file:
        csvReader = csv.reader(file)
        next(csvReader)  # skip header row
        for row in csvReader:
            if not row:
                continue
            if len(row) >= 2:
                dataPoints.append([float(row[0]), float(row[1])])
    return np.array(dataPoints)  # return as numpy array for easier math


# Calculate Euclidean distance manually between two points
def euclideanDistance(pointA, pointB):
    distSquared = 0
    for i in range(len(pointA)):
        distSquared += (pointA[i] - pointB[i]) ** 2
    return distSquared**0.5  # sqrt(sum of squared differences)


# Find neighbors of a point within epsilon distance
def findNeighborPoints(allPoints, currentIndex, epsilon):
    neighborIndices = []
    currentPoint = allPoints[currentIndex]
    for index, otherPoint in enumerate(allPoints):
        if euclideanDistance(currentPoint, otherPoint) <= epsilon:
            neighborIndices.append(index)
    return neighborIndices


# Expand a cluster from a starting core point
def expandCluster(allPoints, clusterLabels, startIndex, clusterId, epsilon, minPoints):
    neighborIndices = findNeighborPoints(allPoints, startIndex, epsilon)
    if len(neighborIndices) < minPoints:  # not a core point, mark as noise
        clusterLabels[startIndex] = -1
        return False
    clusterLabels[startIndex] = clusterId  # assign starting point to cluster
    neighborQueue = set(neighborIndices)
    neighborQueue.discard(startIndex)
    neighborList = list(neighborQueue)

    i = 0
    while i < len(neighborList):
        neighborIndex = neighborList[i]
        if clusterLabels[neighborIndex] in [0, -1]:  # unvisited or noise
            newNeighborIndices = findNeighborPoints(allPoints, neighborIndex, epsilon)
            if len(newNeighborIndices) >= minPoints:  # if neighbor is a core point
                for newIndex in newNeighborIndices:
                    if newIndex not in neighborList and newIndex != startIndex:
                        neighborList.append(newIndex)
            clusterLabels[neighborIndex] = clusterId  # assign neighbor to cluster
        i += 1
    return True


# Run dbsan clustering algorithm
def performDbscanClustering(allPoints, epsilon, minPoints):
    clusterLabels = [0] * len(allPoints)  # 0 = unassigned
    currentClusterId = 0
    for pointIndex in range(len(allPoints)):
        if clusterLabels[pointIndex] != 0:
            continue
        if expandCluster(
            allPoints,
            clusterLabels,
            pointIndex,
            currentClusterId + 1,
            epsilon,
            minPoints,
        ):
            currentClusterId += 1
    return clusterLabels, currentClusterId


# optics clustering class
class OPTICS:
    def __init__(self, minSamples=3, maxEps=np.inf):
        self.minSamples = minSamples
        self.maxEps = maxEps
        self.labels = None
        self.ordering = None
        self.reachability = None

    # manual Euclidean distance for optics
    def euclideanDistance(self, point1, point2):
        distSquared = 0
        for i in range(len(point1)):
            distSquared += (point1[i] - point2[i]) ** 2
        return distSquared**0.5

    # find neighbors and their distances
    def _getNeighborsAndDistances(self, X, pointIndex):
        neighbors = []
        point = X[pointIndex]
        for index in range(len(X)):
            if index != pointIndex:
                distance = self.euclideanDistance(point, X[index])
                if distance <= self.maxEps:
                    neighbors.append((index, distance))
        neighbors.sort(key=lambda x: x[1])  # sort neighbors by distance
        return neighbors

    # distance to (minSamples-1)th neighbor, None if not core point
    def _coreDistance(self, neighbors):
        if len(neighbors) < self.minSamples - 1:
            return None
        return neighbors[self.minSamples - 2][1]

    # calculate reachability distance
    def _reachabilityDistance(self, coreDist, neighborDist):
        if coreDist is None:
            return None
        return max(coreDist, neighborDist)

    # main fit function
    def fit(self, X):
        numSamples = len(X)
        processed = np.zeros(numSamples, dtype=bool)  # track points already processed
        reachability = np.full(numSamples, np.inf)  # reachability distances
        ordering = []  # order points are processed
        seeds = []  # priority queue

        def updateSeeds(seeds, neighbors, coreDist, reachability):
            for neighborIndex, neighborDist in neighbors:
                if not processed[neighborIndex]:
                    newReachDist = self._reachabilityDistance(coreDist, neighborDist)
                    if (
                        newReachDist is not None
                        and newReachDist < reachability[neighborIndex]
                    ):
                        reachability[neighborIndex] = newReachDist
                        heapq.heappush(seeds, (newReachDist, neighborIndex))

        for pointIndex in range(numSamples):
            if processed[pointIndex]:
                continue
            neighbors = self._getNeighborsAndDistances(X, pointIndex)
            processed[pointIndex] = True
            ordering.append(pointIndex)
            coreDist = self._coreDistance(neighbors)
            if coreDist is not None:
                updateSeeds(seeds, neighbors, coreDist, reachability)
                while seeds:  # process points in order of reachability
                    currentReachDist, currentIndex = heapq.heappop(seeds)
                    if processed[currentIndex]:
                        continue
                    neighbors = self._getNeighborsAndDistances(X, currentIndex)
                    processed[currentIndex] = True
                    ordering.append(currentIndex)
                    coreDist = self._coreDistance(neighbors)
                    if coreDist is not None:
                        updateSeeds(seeds, neighbors, coreDist, reachability)

        self.ordering = np.array(ordering)
        self.reachability = reachability
        self.labels = self._extractClusters()  # assign cluster labels
        return self

    # extract clusters from reachability distances
    def _extractClusters(self, eps=None):
        numSamples = len(self.ordering)
        labels = np.full(numSamples, -1)  # default -1 = noise
        if eps is None:
            finiteReach = self.reachability[self.reachability != np.inf]
            if len(finiteReach) == 0:
                return labels
            eps = np.percentile(finiteReach, 70)  # choose threshold automatically
        clusterId = 0
        inCluster = False
        for pointIndex in self.ordering:
            reachValue = self.reachability[pointIndex]
            if reachValue <= eps:
                if not inCluster:
                    clusterId += 1
                    inCluster = True
                labels[pointIndex] = clusterId - 1
            else:
                inCluster = False
                labels[pointIndex] = -1
        return labels


if __name__ == "__main__":
    try:
        dataPoints = loadCsvFile("datagalaxy.csv")  # load data
    except FileNotFoundError:
        print("Error: 'datagalaxy.csv' not found.")
        exit()

    epsilonValue = 5
    minimumPoints = 3

    # Run dbscan
    dbLabels, totalClusters = performDbscanClustering(
        dataPoints, epsilonValue, minimumPoints
    )
    dbClusters = [[] for _ in range(totalClusters)]
    dbNoiseIndices = []
    for index, label in enumerate(dbLabels):
        if label > 0:
            dbClusters[label - 1].append(dataPoints[index])
        elif label == -1:
            dbNoiseIndices.append(index)
    dbClusters = [
        np.array(cluster) for cluster in dbClusters
    ]  # convert to numpy arrays

    clusterColors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "cyan",
        "magenta",
        "yellow",
        "brown",
        "pink",
    ]
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    # Plot dbscan clusters
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    if dbNoiseIndices:
        noisePoints = dataPoints[dbNoiseIndices]
        plt.scatter(
            noisePoints[:, 0],
            noisePoints[:, 1],
            color="black",
            marker="x",
            s=50,
            label="DBSCAN Noise",
        )
    for index, cluster in enumerate(dbClusters):
        if cluster.size > 0:
            color = clusterColors[index % len(clusterColors)]
            plt.scatter(
                cluster[:, 0],
                cluster[:, 1],
                color=color,
                alpha=0.7,
                s=50,
                label=f"Cluster {index+1}",
            )
    plt.title(f"DBSCAN Clusters (Total: {totalClusters})")
    plt.xlabel("Amount Of Stars")
    plt.ylabel("Brightness")
    plt.grid(True, alpha=0.3)

    # plot optics sub-clustering
    plt.subplot(1, 2, 2)
    totalOpticsSubclusters = 0
    if dbNoiseIndices:
        plt.scatter(
            noisePoints[:, 0],
            noisePoints[:, 1],
            color="black",
            marker="x",
            s=50,
            alpha=0.3,
            label="DBSCAN Noise",
        )
    for clusterIndex, cluster in enumerate(dbClusters):
        if cluster.size == 0:
            continue
        baseColor = clusterColors[clusterIndex % len(clusterColors)]
        if len(cluster) < minimumPoints:
            plt.scatter(
                cluster[:, 0],
                cluster[:, 1],
                color=baseColor,
                marker="o",
                s=50,
                alpha=0.5,
            )
            continue
        optics = OPTICS(minSamples=minimumPoints, maxEps=epsilonValue)
        optics.fit(cluster)
        labels = optics.labels
        uniqueLabels = np.unique(labels)
        numSubclusters = len(uniqueLabels) - (1 if -1 in uniqueLabels else 0)
        totalOpticsSubclusters += numSubclusters
        for label in uniqueLabels:
            labelPoints = cluster[labels == label]
            if label == -1:
                plt.scatter(
                    labelPoints[:, 0],
                    labelPoints[:, 1],
                    color=baseColor,
                    marker="x",
                    s=50,
                    alpha=0.5,
                )
            else:
                plt.scatter(
                    labelPoints[:, 0],
                    labelPoints[:, 1],
                    color=baseColor,
                    marker=markers[label % len(markers)],
                    alpha=0.8,
                    s=50,
                    label=f"DBSCAN Cluster {clusterIndex+1} Sub-{label}",
                )
    plt.title(f"OPTICS Sub-clustering (Total Subclusters: {totalOpticsSubclusters})")
    plt.xlabel("Stars")
    plt.ylabel("Brightness")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
