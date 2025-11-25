import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pandas as pd

csv_file = "satellite_data.csv"  # load satellite data from CSV
satellite_data = pd.read_csv(csv_file)

coordinates = satellite_data[["latitude", "longitude"]].values  # extract coordinates
signal_strengths = satellite_data["signal_strength"].values  # extract signal strengths
satellite_names = satellite_data["name"].values  # extract satellite names

p = 4
x_min, x_max = (
    coordinates[:, 0].min() - 2,
    coordinates[:, 0].max() + 2,
)  # this is for plotting limits
y_min, y_max = coordinates[:, 1].min() - 2, coordinates[:, 1].max() + 2

# first the eucclidean case
if p == 2:
    # uses built-in Voronoi and Delaunay functions
    vor = Voronoi(coordinates)
    tri = Delaunay(coordinates)

    fig, ax = plt.subplots(figsize=(10, 10))  # create figure and axis

    # Plot Voronoi edges
    voronoi_plot_2d(
        vor,
        ax=ax,
        show_vertices=False,
        line_colors="blue",
        line_width=1,
        point_size=0,
    )

    # Color Voronoi regions by signal strength
    polygons, colors = [], []
    for i, region_idx in enumerate(vor.point_region):  # iterate over Voronoi regions
        region = vor.regions[region_idx]
        if -1 not in region and region:  # ignore infinite regions
            vertices = vor.vertices[region]
            polygons.append(Polygon(vertices, closed=True))
            colors.append(signal_strengths[i])

    if polygons:  # check if there are any polygons to plot
        collection = PatchCollection(polygons, cmap="viridis", alpha=0.4)
        collection.set_array(np.array(colors))
        ax.add_collection(collection)
        plt.colorbar(collection, ax=ax, label="Signal Strength")

    # Delaunay triangulation
    ax.triplot(
        coordinates[:, 0],
        coordinates[:, 1],
        tri.simplices,
        color="red",
        linewidth=1,
        alpha=0.6,
    )

    ax.set_title("Voronoi + Delaunay (Euclidean, p=2)")
else:
    print("Different norm")

    res = 800  # grid resolution
    x = np.linspace(x_min, x_max, res)  # create grid points
    y = np.linspace(y_min, y_max, res)
    X, Y = np.meshgrid(x, y)  # create meshgrid
    grid = np.stack([X, Y], axis=-1)

    # Compute Lp distances from each grid point to each coordinate
    distances = np.array(
        [np.sum(np.abs(grid - coord) ** p, axis=-1) ** (1 / p) for coord in coordinates]
    )

    # Assign region labels based on minimum distance
    labels = np.argmin(distances, axis=0)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(  # plot the Voronoilike regions
        labels,
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        cmap="tab20",
        alpha=0.4,
        interpolation="nearest",
    )

    # Optional: add Euclidean Delaunay overlay for reference
    tri = Delaunay(coordinates)
    ax.triplot(
        coordinates[:, 0],
        coordinates[:, 1],
        tri.simplices,
        color="red",
        linewidth=1,
        alpha=0.6,
    )

    ax.set_title(f"Simulated Voronoi + Delaunay (L{p}-Norm)")

ax.scatter(coordinates[:, 0], coordinates[:, 1], s=50, c="black", zorder=5)

for i, name in enumerate(satellite_names):  # annotate satellite names
    ax.text(coordinates[i, 0], coordinates[i, 1], f" {name}", fontsize=8)

ax.set_xlabel("Latitude")
ax.set_ylabel("Longitude")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
