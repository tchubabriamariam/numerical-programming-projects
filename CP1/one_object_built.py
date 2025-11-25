import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.cluster import DBSCAN
from scipy.interpolate import UnivariateSpline


MAX_POINTS = 2000
GAUSSIAN_SIGMA = 5.0


def extract_main_object_centroid(
    frame, canny_params, dbscan_params, max_points=MAX_POINTS
):
    # Edge detection with OpenCV
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, canny_params["blur"], 0)
    edges = cv2.Canny(blurred, canny_params["low"], canny_params["high"])

    # Get edge points
    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        return None

    points = np.column_stack((xs, ys))

    # Downsample if too many points
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]

    # Clustering with sklearn's DBSCAN
    clustering = DBSCAN(
        eps=dbscan_params["eps"],
        min_samples=dbscan_params["min_samples"],
        metric="euclidean",
    ).fit(points)

    # Find the largest cluster (main object)
    labels = clustering.labels_
    cluster_ids = [cid for cid in set(labels) if cid != -1]

    if not cluster_ids:
        return None

    # Get the largest cluster
    main_cluster_id = max(cluster_ids, key=lambda c: np.sum(labels == c))
    cluster_points = points[labels == main_cluster_id]

    # Calculate centroid
    centroid = np.mean(cluster_points, axis=0)
    return centroid


def compute_derivatives_numerical(positions, times):
    positions = np.array(positions)
    times = np.array(times)

    if len(positions) < 4:
        return None, None, None, None

    # Compute velocity using numpy's gradient
    vx = np.gradient(positions[:, 0], times)
    vy = np.gradient(positions[:, 1], times)

    # Speed
    speed = np.sqrt(vx**2 + vy**2)

    # acceleration
    accel = np.gradient(speed, times)

    # jerk
    jerk = np.gradient(accel, times)

    # jounce
    jounce = np.gradient(jerk, times)

    return speed, accel, jerk, jounce


def compute_derivatives_savgol(positions, times, window_length=21, polyorder=3):
    positions = np.array(positions)
    times = np.array(times)

    if len(positions) < window_length:
        window_length = (
            len(positions) if len(positions) % 2 == 1 else len(positions) - 1
        )
        if window_length < polyorder + 2:
            return None, None, None, None

    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length -= 1

    try:
        # Get uniform time spacing
        dt = np.mean(np.diff(times))

        # Compute velocity components using Savitzky-Golay filter, googled this
        vx = savgol_filter(positions[:, 0], window_length, polyorder, deriv=1, delta=dt)
        vy = savgol_filter(positions[:, 1], window_length, polyorder, deriv=1, delta=dt)

        # Speed magnitude
        speed = np.sqrt(vx**2 + vy**2)

        # acceleration
        accel = savgol_filter(speed, window_length, polyorder, deriv=1, delta=dt)

        # jerk
        jerk = savgol_filter(speed, window_length, polyorder, deriv=2, delta=dt)

        # jounce
        jounce = savgol_filter(speed, window_length, polyorder, deriv=3, delta=dt)

        return speed, accel, jerk, jounce
    except:
        return None, None, None, None


def compute_derivatives_spline(positions, times, smoothing_factor=100):
    positions = np.array(positions)
    times = np.array(times)

    if len(positions) < 10:
        return None, None, None, None

    # Normalize time for numerical stability
    t_min, t_max = times.min(), times.max()
    time_range = t_max - t_min if t_max > t_min else 1.0
    times_norm = (times - t_min) / time_range if time_range > 0 else times

    try:
        # Fit splines to x and y coordinates
        spline_x = UnivariateSpline(
            times_norm, positions[:, 0], s=smoothing_factor * len(times), k=3
        )
        spline_y = UnivariateSpline(
            times_norm, positions[:, 1], s=smoothing_factor * len(times), k=3
        )

        # Compute velocity components
        vx = spline_x.derivative(1)(times_norm) / time_range
        vy = spline_y.derivative(1)(times_norm) / time_range
        speed = np.sqrt(vx**2 + vy**2)

        # Fit spline to speed for tangential derivatives
        speed_spline = UnivariateSpline(
            times_norm, speed, s=smoothing_factor * len(times), k=3
        )

        # derivatives
        accel = speed_spline.derivative(1)(times_norm) / time_range
        jerk = speed_spline.derivative(2)(times_norm) / (time_range**2)
        jounce = speed_spline.derivative(3)(times_norm) / (time_range**3)

        return speed, accel, jerk, jounce
    except:
        return None, None, None, None


def track_single_object(video_path, canny_params, dbscan_params, frame_skip=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    centroid_history = []
    time_history = []
    frame_count = 0

    print("Starting single object tracking...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Extract centroid
        centroid = extract_main_object_centroid(frame, canny_params, dbscan_params)

        if centroid is not None:
            centroid_history.append(centroid)
            time_history.append(frame_time)

        if frame_count % 100 == 0:
            print(
                f"Processed {frame_count} frames, tracked {len(centroid_history)} centroids"
            )

    cap.release()
    print(
        f"Tracking complete. Total frames: {frame_count}, Centroids: {len(centroid_history)}"
    )

    return np.array(centroid_history), np.array(time_history)


def plot_kinematics_comparison(positions, times, use_smoothing=True, method="all"):
    if len(positions) < 4:
        print("Not enough data points for kinematics computation.")
        return

    # Apply Gaussian smoothing if requested
    if use_smoothing:
        print(f"Applying Gaussian smoothing (sigma={GAUSSIAN_SIGMA})...")
        smoothed_x = gaussian_filter1d(positions[:, 0], sigma=GAUSSIAN_SIGMA)
        smoothed_y = gaussian_filter1d(positions[:, 1], sigma=GAUSSIAN_SIGMA)
        positions = np.column_stack((smoothed_x, smoothed_y))

    # Compute using different methods
    methods_to_plot = []

    if method in ["gradient", "all"]:
        speed_grad, accel_grad, jerk_grad, jounce_grad = compute_derivatives_numerical(
            positions, times
        )
        if speed_grad is not None:
            methods_to_plot.append(
                ("NumPy Gradient", speed_grad, accel_grad, jerk_grad, jounce_grad, "C0")
            )

    if method in ["savgol", "all"]:
        speed_sg, accel_sg, jerk_sg, jounce_sg = compute_derivatives_savgol(
            positions, times
        )
        if speed_sg is not None:
            methods_to_plot.append(
                ("Savitzky-Golay", speed_sg, accel_sg, jerk_sg, jounce_sg, "C1")
            )

    if method in ["spline", "all"]:
        speed_sp, accel_sp, jerk_sp, jounce_sp = compute_derivatives_spline(
            positions, times
        )
        if speed_sp is not None:
            methods_to_plot.append(
                ("Spline", speed_sp, accel_sp, jerk_sp, jounce_sp, "C2")
            )

    if not methods_to_plot:
        print("All computation methods failed.")
        return

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    kinematic_names = [
        "Speed",
        "Tangential Acceleration",
        "Tangential Jerk",
        "Tangential Jounce",
    ]
    ylabels = ["Speed (px/s)", "Accel (px/s²)", "Jerk (px/s³)", "Jounce (px/s⁴)"]

    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    for ax_idx, (name, ylabel) in enumerate(zip(kinematic_names, ylabels)):
        ax = axes_flat[ax_idx]

        for method_name, speed, accel, jerk, jounce, color in methods_to_plot:
            data = [speed, accel, jerk, jounce][ax_idx]
            if data is not None:
                ax.plot(
                    times, data, linewidth=2, label=method_name, color=color, alpha=0.7
                )

        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        if ax_idx == 0:
            ax.set_ylim(bottom=0)
        else:
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    smoothing_text = "with Gaussian Smoothing" if use_smoothing else "without Smoothing"
    plt.suptitle(
        f"Single-Object Kinematics Comparison ({smoothing_text})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


def main():
    video_path = "fredy.mov"

    # Parameters
    canny_params = {"low": 30, "high": 90, "blur": (5, 5)}

    dbscan_params = {"eps": 5, "min_samples": 5}

    # Track the main object
    positions, times = track_single_object(
        video_path=video_path,
        canny_params=canny_params,
        dbscan_params=dbscan_params,
        frame_skip=1,
    )

    if len(positions) < 4:
        print("Not enough tracking data.")
        return

    # 'gradient', 'savgol', 'spline', or 'all' can use this
    plot_kinematics_comparison(positions, times, use_smoothing=True, method="all")


if __name__ == "__main__":
    main()
