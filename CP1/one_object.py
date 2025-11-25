import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline


def euclidean_norm(p1, p2):
    return np.linalg.norm(p1 - p2)


def manhattan_norm(p1, p2):
    return np.sum(np.abs(p1 - p2))


def chebyshev_norm(p1, p2):
    return np.max(np.abs(p1 - p2))


class CannyEdgeDetector:
    def __init__(self, low_threshold=30, high_threshold=90, blur_kernel=(5, 5)):
        self.low = low_threshold
        self.high = high_threshold
        self.blur_kernel = blur_kernel

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        edges = cv2.Canny(blurred, self.low, self.high)
        return edges


class DBSCAN:
    def __init__(self, eps=5.0, min_samples=5, norm_func=euclidean_norm):
        self.eps = eps
        self.min_samples = min_samples
        self.norm_func = norm_func

    def fit(self, points):
        n = len(points)
        if n == 0:
            return np.array([])
        labels = np.full(n, -1)
        visited = np.zeros(n, dtype=bool)
        cid = 0
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self.region(points, i)
            if len(neighbors) < self.min_samples:
                continue
            labels[i] = cid
            self.expand(points, labels, visited, neighbors, cid)
            cid += 1
        return labels

    def region(self, points, idx):
        # Vectorized distance computation for standard norms
        diff = points - points[idx]  # shape (N, 2)
        if self.norm_func == euclidean_norm:
            d = np.linalg.norm(diff, axis=1)
        elif self.norm_func == manhattan_norm:
            d = np.sum(np.abs(diff), axis=1)
        elif self.norm_func == chebyshev_norm:
            d = np.max(np.abs(diff), axis=1)
        else:
            # fallback for arbitrary norm function
            d = np.array([self.norm_func(points[idx], p) for p in points])
        return np.where(d <= self.eps)[0]

    def expand(self, points, labels, visited, neighbors, cid):
        queue = list(neighbors)
        for idx in queue:
            if not visited[idx]:
                visited[idx] = True
                new_neighbors = self.region(points, idx)
                if len(new_neighbors) >= self.min_samples:
                    queue.extend(new_neighbors)
            if labels[idx] == -1:
                labels[idx] = cid


def compute_vector_velocity_numerical(positions, times):
    positions = np.array(positions)
    times = np.array(times)

    if len(positions) < 2:
        return None

    # Velocity using central differences where possible
    vel = np.zeros_like(positions)
    for i in range(len(positions)):
        if i == 0:
            # Forward difference
            dt = times[i + 1] - times[i]
            if dt == 0:
                continue
            vel[i] = (positions[i + 1] - positions[i]) / dt
        elif i == len(positions) - 1:
            # Backward difference
            dt = times[i] - times[i - 1]
            if dt == 0:
                continue
            vel[i] = (positions[i] - positions[i - 1]) / dt
        else:
            # Central difference
            dt = times[i + 1] - times[i - 1]
            if dt == 0:
                continue
            vel[i] = (positions[i + 1] - positions[i - 1]) / dt

    return vel


def compute_scalar_derivatives_numerical(scalar_array, times):
    arr = np.array(scalar_array)
    times = np.array(times)

    if len(arr) < 2:
        return None, None, None

    def diff_array(data, t):
        # Calculates the first derivative using central differences
        deriv = np.zeros_like(data)
        for i in range(len(data)):
            if i == 0:
                dt = t[i + 1] - t[i]
                deriv[i] = (data[i + 1] - data[i]) / dt if dt != 0 else 0
            elif i == len(data) - 1:
                dt = t[i] - t[i - 1]
                deriv[i] = (data[i] - data[i - 1]) / dt if dt != 0 else 0
            else:
                dt = t[i + 1] - t[i - 1]
                deriv[i] = (data[i + 1] - data[i - 1]) / dt if dt != 0 else 0
        return deriv

    # Acceleration
    a_t = diff_array(arr, times)

    # Jerk
    j_t = diff_array(a_t, times)

    # Jounce
    o_t = diff_array(j_t, times)

    return a_t, j_t, o_t


def main(video_path, norm_func=euclidean_norm, max_points=2000, frame_skip=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Cannot open video:", video_path)
        return

    edge = CannyEdgeDetector(30, 90)
    db = DBSCAN(eps=5, min_samples=5, norm_func=norm_func)

    centroid_history = []
    time_history = []

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % frame_skip != 0:
            continue

        edges = edge.detect(frame)
        ys, xs = np.where(edges > 0)
        points = np.column_stack((xs, ys))
        if len(points) == 0:
            continue

        # Downsample points if too many
        if len(points) > max_points:
            idxs = np.random.choice(len(points), max_points, replace=False)
            points = points[idxs]

        labels = db.fit(points)
        cluster_ids = [cid for cid in np.unique(labels) if cid != -1]
        if not cluster_ids:
            continue

        # Pick largest cluster as main object
        main_cid = max(cluster_ids, key=lambda c: np.sum(labels == c))
        cluster_points = points[labels == main_cid]
        cx, cy = np.mean(cluster_points[:, 0]), np.mean(cluster_points[:, 1])

        centroid_history.append((cx, cy))
        time_history.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

    cap.release()
    print(f"Processed {frame_count} frames, found {len(centroid_history)} centroids.")

    if len(centroid_history) < 4:
        print("Not enough centroid data (min 4) to compute derivatives.")
        return

    # Convert to NumPy array for processing
    centroid_array = np.array(centroid_history)

    GAUSSIAN_SIGMA = 5.0
    print(
        f"Applying Gaussian smoothing (sigma={GAUSSIAN_SIGMA}) to raw position data for stable derivatives..."
    )

    # Smooth X and Y components separately
    smoothed_x = gaussian_filter1d(centroid_array[:, 0], sigma=GAUSSIAN_SIGMA)
    smoothed_y = gaussian_filter1d(centroid_array[:, 1], sigma=GAUSSIAN_SIGMA)
    smoothed_positions = np.column_stack((smoothed_x, smoothed_y))

    print("Using Central Difference for Velocity/Speed...")
    # Velocity is calculated using central difference on the smoothed positions
    vel_vec = compute_vector_velocity_numerical(smoothed_positions, time_history)

    if vel_vec is None:
        print("Velocity computation failed.")
        return

    def magnitude(arr):
        if arr is None:
            return None
        return np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)

    speed_mag = magnitude(vel_vec)
    if speed_mag is None:
        print("Speed magnitude is None.")
        return

    print("Using Central Difference successively for A, J, and O...")
    # A, J, and O are calculated using central difference on the speed_mag array
    a_tangential, j_tangential, o_tangential = compute_scalar_derivatives_numerical(
        speed_mag, time_history
    )

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    # Use a flag to indicate the method used
    method_label = "Numerical Central Diff"

    ax[0, 0].plot(time_history, speed_mag, linewidth=2)
    ax[0, 0].set_title(
        f"Speed $|\mathbf{{v}}|$ ({method_label})", fontsize=12, fontweight="bold"
    )
    ax[0, 0].set_xlabel("Time (s)")
    ax[0, 0].set_ylabel("Speed (pixels/sec)")
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].set_ylim(bottom=0)

    if a_tangential is not None:
        ax[0, 1].plot(time_history, a_tangential, linewidth=2, color="orange")
        ax[0, 1].set_title(
            f"Tangential Acceleration $a_{{t}}$ ({method_label})",
            fontsize=12,
            fontweight="bold",
        )
        ax[0, 1].set_xlabel("Time (s)")
        ax[0, 1].set_ylabel("Accel $\\frac{{d|\\mathbf{{v}}|}}{{dt}}$ (pixels/sec²)")
        ax[0, 1].grid(True, alpha=0.3)
        ax[0, 1].axhline(0, color="gray", linestyle="--")
    else:
        ax[0, 1].text(
            0.5,
            0.5,
            "Accel computation failed",
            ha="center",
            va="center",
            transform=ax[0, 1].transAxes,
            fontsize=11,
        )

    if j_tangential is not None:
        ax[1, 0].plot(time_history, j_tangential, linewidth=2, color="green")
        ax[1, 0].set_title(
            f"Tangential Jerk $j_{{t}}$ ({method_label})",
            fontsize=12,
            fontweight="bold",
        )
        ax[1, 0].set_xlabel("Time (s)")
        ax[1, 0].set_ylabel("Jerk $\\frac{{d^2|\\mathbf{{v}}|}}{{dt^2}}$ (pixels/sec³)")
        ax[1, 0].grid(True, alpha=0.3)
        ax[1, 0].axhline(0, color="gray", linestyle="--")
    else:
        ax[1, 0].text(
            0.5,
            0.5,
            "Jerk computation failed",
            ha="center",
            va="center",
            transform=ax[1, 0].transAxes,
            fontsize=11,
        )

    if o_tangential is not None and len(o_tangential) > 0:
        ax[1, 1].plot(time_history, o_tangential, linewidth=2, color="red")
        ax[1, 1].set_title(
            f"Tangential Jounce $o_{{t}}$ ({method_label})",
            fontsize=12,
            fontweight="bold",
        )
        ax[1, 1].set_xlabel("Time (s)")
        ax[1, 1].set_ylabel(
            "Jounce $\\frac{{d^3|\\mathbf{{v}}|}}{{dt^3}}$ (pixels/sec⁴)"
        )
        ax[1, 1].grid(True, alpha=0.3)
        ax[1, 1].axhline(0, color="gray", linestyle="--")
    else:
        ax[1, 1].text(
            0.5,
            0.5,
            "Jounce computation failed",
            ha="center",
            va="center",
            transform=ax[1, 1].transAxes,
            fontsize=11,
        )

    plt.suptitle(
        "Single-Object Kinematics: Pure Central Difference (Requires Position Smoothing)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


if __name__ == "__main__":
    # Central difference is highly sensitive, hence the mandatory Gaussian smoothing.
    main(
        "fredy.mov",
        norm_func=euclidean_norm,
        max_points=2000,
        frame_skip=1,
    )
