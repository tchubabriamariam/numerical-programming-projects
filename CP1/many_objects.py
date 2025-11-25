import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

MAX_POINTS = 2000
MIN_TRACK_LENGTH = 10
MAX_MISSING_FRAMES = 10
ALPHA = 0.5


# i dont use this
def manhattan_norm(p1, p2):
    return np.sum(np.abs(p1 - p2))


def euclidean_norm(p1, p2):
    return np.linalg.norm(p1 - p2)


def kinematic_norm(p1_state, p2_state, alpha=ALPHA):
    # Position difference
    pos_diff_sq = (p1_state[0] - p2_state[0]) ** 2 + (p1_state[1] - p2_state[1]) ** 2

    # Velocity difference weighted by alpha**2
    vel_diff_sq = (p1_state[2] - p2_state[2]) ** 2 + (p1_state[3] - p2_state[3]) ** 2

    # Kinematic Distance, as i googled this is good
    d_kinematic = np.sqrt(pos_diff_sq + (alpha**2) * vel_diff_sq)
    return d_kinematic


class ObjectTracker:
    def __init__(self, initial_centroid, frame_time, tracker_id):
        self.id = tracker_id
        # Standard lists to store ALL history for final plotting
        self.centroid_history = [np.array(initial_centroid)]
        self.time_history = [frame_time]
        self.missing_frames = 0
        self.is_active = True
        self.last_velocity = np.array([0.0, 0.0])

    def add_position(self, new_centroid, frame_time):
        self.centroid_history.append(np.array(new_centroid))
        self.time_history.append(frame_time)
        self.missing_frames = 0
        self._update_velocity()

    def _update_velocity(self):
        if len(self.centroid_history) >= 2:
            dt = self.time_history[-1] - self.time_history[-2]
            if dt > 0:
                dx = self.centroid_history[-1] - self.centroid_history[-2]
                self.last_velocity = dx / dt

    def get_kinematic_state(self):
        last_pos = self.centroid_history[-1]
        return np.array(
            [last_pos[0], last_pos[1], self.last_velocity[0], self.last_velocity[1]]
        )

    def predict_next_position(self):
        # using last parttion
        return self.centroid_history[-1]

    def mark_missing(self):
        self.missing_frames += 1
        if self.missing_frames > MAX_MISSING_FRAMES:
            self.is_active = False


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


# dbscan from ap
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
        diff = points - points[idx]
        d = np.linalg.norm(diff, axis=1)
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


def compute_scalar_derivative(scalar_array, times, order, smoothing_factor=100):
    if len(scalar_array) < 10:
        return None

    times = np.array(times)
    scalar_array = np.array(scalar_array)

    t_min, t_max = times.min(), times.max()
    times_norm = (times - t_min) / (t_max - t_min) if t_max > t_min else times
    s_value = smoothing_factor * len(times)
    time_scale = (t_max - t_min) if t_max > t_min else 1.0

    try:
        # k=3?
        spline = UnivariateSpline(times_norm, scalar_array, s=s_value, k=3)
    except:
        return None

    # Scale factor for the derivative based on the order
    scale_factor = time_scale**order

    # Calculate the derivative and scale it back to real time
    derivative = spline.derivative(order)(times_norm) / scale_factor

    return derivative


def compute_derivatives_spline(positions, times, smoothing_factor=100):
    positions = np.array(positions)
    times = np.array(times)

    if len(positions) < 10:
        return None, None, None, None

    t_min, t_max = times.min(), times.max()
    times_norm = (times - t_min) / (t_max - t_min) if t_max > t_min else times
    s_value = smoothing_factor * len(times)
    time_scale = (t_max - t_min) if t_max > t_min else 1.0

    k_degree = 3
    try:
        spline_x = UnivariateSpline(times_norm, positions[:, 0], s=s_value, k=k_degree)
        spline_y = UnivariateSpline(times_norm, positions[:, 1], s=s_value, k=k_degree)
    except:
        return None, None, None, None

    vel_x = spline_x.derivative(1)(times_norm) / time_scale
    vel_y = spline_y.derivative(1)(times_norm) / time_scale
    vel = np.column_stack((vel_x, vel_y))

    # not using
    acc = None
    jerk = None
    jounce = None

    return vel, acc, jerk, jounce


def magnitude(arr):
    if arr is None:
        return None
    return np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)


def main_multi(
    video_path,
    # Canny parameters
    canny_low=30,
    canny_high=90,
    canny_blur=(5, 5),
    # DBSCAN parameters
    eps_dbscan=5,
    min_samples_dbscan=5,
    # Tracker parameters
    association_radius=50,
    frame_skip=1,
    use_spline=True,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Cannot open video:", video_path)
        return

    edge = CannyEdgeDetector(
        low_threshold=canny_low, high_threshold=canny_high, blur_kernel=canny_blur
    )
    db = DBSCAN(
        eps=eps_dbscan, min_samples=min_samples_dbscan, norm_func=euclidean_norm
    )

    active_trackers = []
    next_tracker_id = 0
    frame_count = 0

    # tracking loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % frame_skip != 0:
            continue

        frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Edge Detection & Clustering
        edges = edge.detect(frame)
        ys, xs = np.where(edges > 0)
        points = np.column_stack((xs, ys))

        if len(points) == 0:
            for tracker in active_trackers:
                tracker.mark_missing()
            active_trackers = [t for t in active_trackers if t.is_active]
            continue

        if len(points) > MAX_POINTS:
            idxs = np.random.choice(len(points), MAX_POINTS, replace=False)
            points = points[idxs]

        labels = db.fit(points)
        cluster_ids = [cid for cid in np.unique(labels) if cid != -1]

        current_centroids = {}
        for cid in cluster_ids:
            cluster_points = points[labels == cid]
            cx, cy = np.mean(cluster_points[:, 0]), np.mean(cluster_points[:, 1])

            if len(cluster_points) >= min_samples_dbscan:
                current_centroids[cid] = np.array([cx, cy])

        # Data Association
        unmatched_centroids = list(current_centroids.values())

        for tracker in list(active_trackers):
            if not tracker.is_active:
                continue

            if len(tracker.centroid_history) >= 2:
                tracker_state = tracker.get_kinematic_state()
                distances = []

                for pos in unmatched_centroids:
                    mock_cluster_state = np.array(
                        [pos[0], pos[1], tracker_state[2], tracker_state[3]]
                    )
                    distances.append(kinematic_norm(tracker_state, mock_cluster_state))
            else:
                predicted_pos = tracker.predict_next_position()
                distances = [
                    euclidean_norm(predicted_pos, c) for c in unmatched_centroids
                ]

            if distances:
                min_dist_idx = np.argmin(distances)
                min_dist = distances[min_dist_idx]

                if min_dist < association_radius:
                    new_centroid = unmatched_centroids.pop(min_dist_idx)
                    tracker.add_position(new_centroid, frame_time)
                else:
                    tracker.mark_missing()
            else:
                tracker.mark_missing()

        # create new tracker
        for new_centroid in unmatched_centroids:
            new_tracker = ObjectTracker(new_centroid, frame_time, next_tracker_id)
            active_trackers.append(new_tracker)
            next_tracker_id += 1

        # filter lost trackers
        active_trackers = [t for t in active_trackers if t.is_active]

    cap.release()
    print(f"Processed {frame_count} frames. Total objects tracked: {next_tracker_id}")

    valid_tracks = [
        t for t in active_trackers if len(t.centroid_history) >= MIN_TRACK_LENGTH
    ]

    if not valid_tracks:
        print(
            "No tracks long enough to compute motion (min length is %d)."
            % MIN_TRACK_LENGTH
        )
        return

    print("\n--- Kinematic Norm Association Analysis (Alpha = %.2f) ---" % ALPHA)

    num_tracks = len(valid_tracks)
    fig, axes = plt.subplots(num_tracks, 4, figsize=(18, 4 * num_tracks), squeeze=False)

    print(f"Generating tangential kinematic plots for {num_tracks} tracks.")

    # Consistent smoothing factor for all derivatives
    SMOOTH_FACTOR = 10000

    for i, tracker in enumerate(valid_tracks):
        pos = tracker.centroid_history
        time = tracker.time_history

        # Compute vector velocity to get Speed Magnitude
        vel_vec, _, _, _ = compute_derivatives_spline(
            pos,
            time,
            smoothing_factor=SMOOTH_FACTOR,
        )

        speed_mag = magnitude(vel_vec)

        # Initialize tangential kinematics
        a_tangential = None
        j_tangential = None
        o_tangential = None

        if speed_mag is not None:
            # acccseleration
            a_tangential = compute_scalar_derivative(
                speed_mag, time, order=1, smoothing_factor=SMOOTH_FACTOR
            )

            # jerk
            j_tangential = compute_scalar_derivative(
                speed_mag, time, order=2, smoothing_factor=SMOOTH_FACTOR
            )

            # jounce
            o_tangential = compute_scalar_derivative(
                speed_mag, time, order=3, smoothing_factor=SMOOTH_FACTOR
            )

        if speed_mag is not None:
            axes[i, 0].plot(time, speed_mag, linewidth=1.5, color="C0")
            axes[i, 0].set_title(
                f"Speed $|\mathbf{{v}}|$ (Object {tracker.id})", fontsize=10
            )
            axes[i, 0].set_ylabel("Speed (px/s)", fontsize=8)
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].set_ylim(bottom=0)

        if a_tangential is not None:
            axes[i, 1].plot(time, a_tangential, linewidth=1.5, color="C1")
            axes[i, 1].set_title(
                f"Tangential Accel $a_{{t}}$ (Object {tracker.id})", fontsize=10
            )
            axes[i, 1].set_ylabel(
                "Accel $\\frac{{d|\\mathbf{{v}}|}}{{dt}}$ (px/s²)", fontsize=8
            )
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].axhline(0, color="gray", linestyle="--")

        if j_tangential is not None:
            axes[i, 2].plot(time, j_tangential, linewidth=1.5, color="C2")
            axes[i, 2].set_title(
                f"Tangential Jerk $j_{{t}}$ (Object {tracker.id})", fontsize=10
            )
            axes[i, 2].set_ylabel(
                "Jerk $\\frac{{d^2|\\mathbf{{v}}|}}{{dt^2}}$ (px/s³)", fontsize=8
            )
            axes[i, 2].grid(True, alpha=0.3)
            axes[i, 2].axhline(0, color="gray", linestyle="--")

        if o_tangential is not None:
            axes[i, 3].plot(time, o_tangential, linewidth=1.5, color="C3")
            axes[i, 3].set_title(
                f"Tangential Jounce $o_{{t}}$ (Object {tracker.id})", fontsize=10
            )
            axes[i, 3].set_ylabel(
                "Jounce $\\frac{{d^3|\\mathbf{{v}}|}}{{dt^3}}$ (px/s⁴)", fontsize=8
            )
            axes[i, 3].grid(True, alpha=0.3)
            axes[i, 3].axhline(0, color="gray", linestyle="--")

        if i == num_tracks - 1:
            axes[i, 0].set_xlabel("Time (s)", fontsize=8)
            axes[i, 1].set_xlabel("Time (s)", fontsize=8)
            axes[i, 2].set_xlabel("Time (s)", fontsize=8)
            axes[i, 3].set_xlabel("Time (s)", fontsize=8)

    plt.suptitle(
        "Multi-Object Kinematic Analysis (Pure Tangential Kinematics)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


if __name__ == "__main__":
    main_multi(
        "giraffe.mp4",
        # Canny settings: Keep these sensitive settings.
        canny_low=20,
        canny_high=60,
        canny_blur=(3, 3),
        # dbscan
        eps_dbscan=20,
        min_samples_dbscan=34,  # this is 24th try
        association_radius=120,
        frame_skip=1,
        use_spline=True,
    )
