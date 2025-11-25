import numpy as np
import matplotlib.pyplot as plt
import csv


def loadCsvFile(filePath):
    dataPoints = []
    with open(filePath, "r") as file:
        csvReader = csv.reader(file)
        next(csvReader)
        for row in csvReader:
            if not row:
                continue
            dataPoints.append([float(row[0]), float(row[1])])
    return np.array(dataPoints)


dataPoints = loadCsvFile("datagalaxy.csv")
print("Loaded points:", dataPoints.shape)


# euclidean distance calculation
def calculateEuclideanDistance(pointA, pointB):
    return sum((pointA[i] - pointB[i]) ** 2 for i in range(len(pointA))) ** 0.5


# first norm distance calculation
def calculateNormOne(pointA, pointB):
    return sum(abs(pointA[i] - pointB[i]) for i in range(len(pointA)))


# find neighbor points within epsilon distance
def findNeighborPoints(allPoints, currentPointIndex, epsilon):
    neighborIndices = []
    for index, otherPoint in enumerate(allPoints):
        if (
            calculateEuclideanDistance(allPoints[currentPointIndex], otherPoint)
            <= epsilon
        ):
            neighborIndices.append(index)
    return neighborIndices


# expand cluster from a seed point
def expandWholeCluster(
    allPoints, clusterLabels, startPointIndex, clusterId, epsilon, minPoints
):
    neighborIndices = findNeighborPoints(allPoints, startPointIndex, epsilon)
    if len(neighborIndices) < minPoints:
        clusterLabels[startPointIndex] = -1
        return False
    clusterLabels[startPointIndex] = clusterId

    i = 0
    while i < len(neighborIndices):
        neighborIndex = neighborIndices[i]
        if clusterLabels[neighborIndex] == -1:
            clusterLabels[neighborIndex] = clusterId
        elif clusterLabels[neighborIndex] == 0:
            newNeighborIndices = findNeighborPoints(allPoints, neighborIndex, epsilon)
            if len(newNeighborIndices) < minPoints:
                clusterLabels[neighborIndex] = -1
            else:
                clusterLabels[neighborIndex] = clusterId
                for newIndex in newNeighborIndices:
                    if newIndex not in neighborIndices:
                        neighborIndices.append(newIndex)
        i += 1
    return True


# main dbscan function
def performDbscanClustering(allPoints, epsilon, minPoints):
    clusterLabels = [0] * len(allPoints)
    currentClusterId = 0
    for pointIndex in range(len(allPoints)):
        if clusterLabels[pointIndex] != 0:
            continue
        if expandWholeCluster(
            allPoints,
            clusterLabels,
            pointIndex,
            currentClusterId + 1,
            epsilon,
            minPoints,
        ):
            currentClusterId += 1
    return clusterLabels, currentClusterId


# run dbscan
epsilonValue = 5
minimumPoints = 3
clusterLabels, totalClusters = performDbscanClustering(
    dataPoints, epsilonValue, minimumPoints
)
print(clusterLabels)

# drawing
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
plt.figure(figsize=(8, 6))

for i, point in enumerate(dataPoints):
    if clusterLabels[i] == -1:
        plt.scatter(point[0], point[1], color="black", marker="x")  # noise
    else:
        plt.scatter(
            point[0],
            point[1],
            color=clusterColors[(clusterLabels[i] - 1) % len(clusterColors)],
            alpha=0.7,
        )

plt.title(f"DBSCAN Clustering (clusters = {totalClusters})")
plt.xlabel("Amount of stars")
plt.ylabel("Brightness of galaxy")
plt.show()
