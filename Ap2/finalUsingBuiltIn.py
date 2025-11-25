import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.cluster import DBSCAN, OPTICS


# used geeksforgeeks for this
def loadCsvFile(filePath):
    dataPoints = []
    with open(filePath, "r") as file:
        csvReader = csv.reader(file)
        next(csvReader)  # Skip header
        for row in csvReader:
            if not row:
                continue
            dataPoints.append([float(row[0]), float(row[1])])
    return np.array(dataPoints)


X = loadCsvFile("datagalaxy.csv")
# dbscan
db = DBSCAN(eps=5, min_samples=3).fit(X)
labels_db = db.labels_
core_samples_mask = np.zeros_like(labels_db, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

unique_labels_db = set(labels_db)
colors = ["y", "b", "g", "r", "c", "m", "orange", "purple"]
markers_optics = ["o", "s", "^", "D", "v", "<", ">"]

plt.figure(figsize=(10, 8))

for k, col in zip(unique_labels_db, colors):
    class_member_mask = labels_db == k
    xy_core = X[class_member_mask & core_samples_mask]
    xy_border = X[class_member_mask & ~core_samples_mask]

    if k == -1:
        col = "k"
        plt.scatter(xy_core[:, 0], xy_core[:, 1], c=col, edgecolor="k", s=50)
        plt.scatter(
            xy_border[:, 0], xy_border[:, 1], c=col, edgecolor="k", s=50, alpha=0.5
        )
        continue

    plt.scatter(xy_core[:, 0], xy_core[:, 1], c=col, edgecolor="k", s=50)
    plt.scatter(xy_border[:, 0], xy_border[:, 1], c=col, edgecolor="k", s=50, alpha=0.5)

    cluster_size = len(X[class_member_mask])
    if cluster_size > 1:
        min_samples_optics = min(5, cluster_size)
        # optics
        optics = OPTICS(min_samples=min_samples_optics, xi=0.05, min_cluster_size=0.05)
        optics.fit(X[class_member_mask])
        labels_optics = optics.labels_
        unique_labels_optics = set(labels_optics)

        for i, sub_label in enumerate(unique_labels_optics):
            sub_mask = labels_optics == sub_label
            marker = "x" if sub_label == -1 else markers_optics[i % len(markers_optics)]
            plt.scatter(
                X[class_member_mask][sub_mask, 0],
                X[class_member_mask][sub_mask, 1],
                c=col,
                marker=marker,
                edgecolor="k",
                s=100,
                alpha=0.8,
            )

plt.title("DBSCAN + OPTICS Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
