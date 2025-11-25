import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from filterpy.kalman import KalmanFilter
from collections import defaultdict


MAX_POINTS = 2000
MIN_TRACK_LENGTH = 50  # Increased to filter out short spurious tracks
MAX_MISSING_FRAMES = 15  # Increased to allow objects to persist longer
ALPHA = 0.5  # Velocity weighting for kinematic distance


class KalmanTracker:
    def __init__(self, initial_position, frame_time, tracker_id):
        self.id = tracker_id
        self.position_history = [np.array(initial_position)]
        self.time_history = [frame_time]
        self.missing_frames = 0
        self.is_active = True

        # Initialize Kalman Filter for 2D position + velocity
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # State transition matrix (constant velocity model)
        dt = 1 / 30  # assume 30fps initially
        self.kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Measurement function (observe position only)
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # Covariance matrices
        self.kf.R *= 5  # measurement noise
        self.kf.P *= 100  # initial uncertainty
        self.kf.Q *= 0.1  # process noise

        # Initialize state
        self.kf.x = np.array([initial_position[0], initial_position[1], 0, 0])

    def predict(self):
        self.kf.predict()
        return self.kf.x[:2]  # return predicted position

    def update(self, measurement, frame_time):
        self.kf.update(measurement)
        self.position_history.append(np.array(measurement))
        self.time_history.append(frame_time)
        self.missing_frames = 0

        # Update dt for next prediction
        if len(self.time_history) >= 2:
            dt = self.time_history[-1] - self.time_history[-2]
            self.kf.F[0, 2] = dt
            self.kf.F[1, 3] = dt

    def get_velocity(self):
        return self.kf.x[2:4]

    def get_state(self):
        return self.kf.x.copy()

    def mark_missing(self):
        self.missing_frames += 1
        if self.missing_frames > MAX_MISSING_FRAMES:
            self.is_active = False


def kinematic_distance(state1, state2, alpha=ALPHA):
    pos_dist = np.sqrt((state1[0] - state2[0]) ** 2 + (state1[1] - state2[1]) ** 2)
    vel_dist = np.sqrt((state1[2] - state2[2]) ** 2 + (state1[3] - state2[3]) ** 2)
    return np.sqrt(pos_dist**2 + (alpha * vel_dist) ** 2)


def extract_edge_centroids(frame, canny_params, dbscan_params, max_points=MAX_POINTS):
    # Edge detection with OpenCV
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, canny_params["blur"], 0)
    edges = cv2.Canny(blurred, canny_params["low"], canny_params["high"])

    # Get edge points
    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        return []

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

    # Calculate centroids for each cluster
    centroids = []
    for label in set(clustering.labels_):
        if label == -1:  # skip noise
            continue
        cluster_points = points[clustering.labels_ == label]
        if len(cluster_points) >= dbscan_params["min_samples"]:
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)

    return centroids


def compute_tangential_kinematics(positions, times, smoothing_factor=10000):
    if len(positions) < 10:
        return None, None, None, None

    positions = np.array(positions)
    times = np.array(times)

    # Normalize time for numerical stability
    t_min, t_max = times.min(), times.max()
    time_range = t_max - t_min if t_max > t_min else 1.0
    times_norm = (times - t_min) / time_range if time_range > 0 else times

    # Fit splines to x and y coordinates
    try:
        spline_x = UnivariateSpline(
            times_norm, positions[:, 0], s=smoothing_factor * len(times), k=3
        )
        spline_y = UnivariateSpline(
            times_norm, positions[:, 1], s=smoothing_factor * len(times), k=3
        )
    except:
        return None, None, None, None

    # Compute velocity components
    vx = spline_x.derivative(1)(times_norm) / time_range
    vy = spline_y.derivative(1)(times_norm) / time_range
    speed = np.sqrt(vx**2 + vy**2)

    # Fit spline to speed magnitude for tangential derivatives
    try:
        speed_spline = UnivariateSpline(
            times_norm, speed, s=smoothing_factor * len(times), k=3
        )
    except:
        return speed, None, None, None

    # Tangential acceleration (d|v|/dt)
    accel = speed_spline.derivative(1)(times_norm) / time_range

    # Tangential jerk (d²|v|/dt²)
    jerk = speed_spline.derivative(2)(times_norm) / (time_range**2)

    # Tangential jounce (d³|v|/dt³)
    jounce = speed_spline.derivative(3)(times_norm) / (time_range**3)

    return speed, accel, jerk, jounce


def track_objects(
    video_path, canny_params, dbscan_params, association_radius=120, frame_skip=1
):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    active_trackers = []
    next_id = 0
    frame_count = 0

    print("Starting tracking...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Extract centroids from current frame
        centroids = extract_edge_centroids(frame, canny_params, dbscan_params)

        if len(centroids) == 0:
            for tracker in active_trackers:
                tracker.mark_missing()
            active_trackers = [t for t in active_trackers if t.is_active]
            continue

        # Predict next positions for all trackers
        predictions = []
        for tracker in active_trackers:
            if tracker.is_active:
                pred = tracker.predict()
                predictions.append(pred)
            else:
                predictions.append(None)

        # Data association using Hungarian algorithm via scipy
        if active_trackers:
            # Build cost matrix using kinematic distance
            cost_matrix = np.zeros((len(active_trackers), len(centroids)))
            for i, tracker in enumerate(active_trackers):
                if not tracker.is_active or predictions[i] is None:
                    cost_matrix[i, :] = 1e6
                    continue

                tracker_state = tracker.get_state()
                for j, centroid in enumerate(centroids):
                    # Create pseudo-state for centroid (assume same velocity)
                    centroid_state = np.array(
                        [centroid[0], centroid[1], tracker_state[2], tracker_state[3]]
                    )
                    cost_matrix[i, j] = kinematic_distance(
                        tracker_state, centroid_state
                    )

            # Use scipy's linear assignment (Hungarian algorithm)
            from scipy.optimize import linear_sum_assignment

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_centroids = set()
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < association_radius:
                    active_trackers[i].update(centroids[j], frame_time)
                    matched_centroids.add(j)
                else:
                    active_trackers[i].mark_missing()

            # Create new trackers for unmatched centroids
            for j, centroid in enumerate(centroids):
                if j not in matched_centroids:
                    new_tracker = KalmanTracker(centroid, frame_time, next_id)
                    active_trackers.append(new_tracker)
                    next_id += 1
        else:
            # No active trackers, create new ones
            for centroid in centroids:
                new_tracker = KalmanTracker(centroid, frame_time, next_id)
                active_trackers.append(new_tracker)
                next_id += 1

        # Remove inactive trackers
        active_trackers = [t for t in active_trackers if t.is_active]

        if frame_count % 100 == 0:
            print(
                f"Processed {frame_count} frames, active trackers: {len(active_trackers)}"
            )

    cap.release()
    print(f"Tracking complete. Total frames: {frame_count}, Total objects: {next_id}")

    return active_trackers


def plot_kinematics(trackers, min_length=MIN_TRACK_LENGTH):
    valid_tracks = [t for t in trackers if len(t.position_history) >= min_length]

    if not valid_tracks:
        print(f"No tracks with at least {min_length} points.")
        return

    num_tracks = len(valid_tracks)
    fig, axes = plt.subplots(num_tracks, 4, figsize=(18, 4 * num_tracks), squeeze=False)

    print(f"Plotting kinematics for {num_tracks} tracks...")

    for i, tracker in enumerate(valid_tracks):
        pos = tracker.position_history
        time = tracker.time_history

        # Compute tangential kinematics
        speed, accel, jerk, jounce = compute_tangential_kinematics(pos, time)

        # Plot speed
        if speed is not None:
            axes[i, 0].plot(time, speed, linewidth=1.5, color="C0")
            axes[i, 0].set_title(f"Speed |v| (Object {tracker.id})", fontsize=10)
            axes[i, 0].set_ylabel("Speed (px/s)", fontsize=8)
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].set_ylim(bottom=0)

        # Plot tangential acceleration
        if accel is not None:
            axes[i, 1].plot(time, accel, linewidth=1.5, color="C1")
            axes[i, 1].set_title(f"Tangential Accel (Object {tracker.id})", fontsize=10)
            axes[i, 1].set_ylabel("Accel (px/s²)", fontsize=8)
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].axhline(0, color="gray", linestyle="--", alpha=0.5)

        # Plot tangential jerk
        if jerk is not None:
            axes[i, 2].plot(time, jerk, linewidth=1.5, color="C2")
            axes[i, 2].set_title(f"Tangential Jerk (Object {tracker.id})", fontsize=10)
            axes[i, 2].set_ylabel("Jerk (px/s³)", fontsize=8)
            axes[i, 2].grid(True, alpha=0.3)
            axes[i, 2].axhline(0, color="gray", linestyle="--", alpha=0.5)

        # Plot tangential jounce
        if jounce is not None:
            axes[i, 3].plot(time, jounce, linewidth=1.5, color="C3")
            axes[i, 3].set_title(
                f"Tangential Jounce (Object {tracker.id})", fontsize=10
            )
            axes[i, 3].set_ylabel("Jounce (px/s⁴)", fontsize=8)
            axes[i, 3].grid(True, alpha=0.3)
            axes[i, 3].axhline(0, color="gray", linestyle="--", alpha=0.5)

        # X-labels for bottom row
        if i == num_tracks - 1:
            for col in range(4):
                axes[i, col].set_xlabel("Time (s)", fontsize=8)

    plt.suptitle(
        "Multi-Object Kinematic Analysis (Using Standard Libraries)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


def main():
    video_path = "giraffe.mp4"

    # Parameters - tuned to be more conservative
    canny_params = {"low": 30, "high": 80, "blur": (5, 5)}

    dbscan_params = {
        "eps": 25,
        "min_samples": 50,  # Increased to require larger clusters
    }

    # Track objects
    trackers = track_objects(
        video_path=video_path,
        canny_params=canny_params,
        dbscan_params=dbscan_params,
        association_radius=80,  # Tighter association
        frame_skip=1,
    )

    # Plot results
    plot_kinematics(trackers)


if __name__ == "__main__":
    main()
