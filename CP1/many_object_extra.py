import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import savgol_filter

# Configuration parameters for our motion detection and tracking
video = "giraffe.mp4"
threshold = 3  # Minimum brightness difference to count as motion
blob_area = 700  # Minimum size for detected motion regions
eps = 50  # Distance threshold for clustering points
dbscan_min = 1  # Minimum points to form a cluster
widnow = 7  # Window size for smoothing motion tracks
kernel = 3  # Kernel size for noise removal
dilate = 2  # Additional dilation to connect nearby motion regions


def to_gray(img):
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)


def abs_diff(a, b):
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)


def binarize(img, thr):
    return (img > thr).astype(np.uint8)


def morph_open(mask, k):
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def connected_components(mask):
    H, W = mask.shape
    visited = np.zeros_like(mask, bool)  # Track visited pixels
    components = []  # Store detected components

    # Get coordinates of all white pixels in the mask
    y_idxs, x_idxs = np.nonzero(mask)
    if len(y_idxs) == 0:
        return []  # No motion detected

    # Define 4-connected neighborhood (up, down, left, right)
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # Process each white pixel that hasn't been visited
    for y0, x0 in zip(y_idxs, x_idxs):
        if visited[y0, x0]:
            continue

        # Start BFS from this pixel
        queue = deque()
        queue.append((y0, x0))
        visited[y0, x0] = 1
        pixels = []  # Store all pixels in this component

        # Track bounding box coordinates
        minx = maxx = x0
        miny = maxy = y0

        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))

            # Update bounding box
            minx = min(minx, x)
            maxx = max(maxx, x)
            miny = min(miny, y)
            maxy = max(maxy, y)

            # Check all neighbors
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                # If neighbor is within image, is white, and not visited
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = 1
                    queue.append((ny, nx))

        # Calculate component properties
        area = len(pixels)
        cy = int(np.mean([p[0] for p in pixels]))  # Center y
        cx = int(np.mean([p[1] for p in pixels]))  # Center x

        components.append(
            {
                "area": area,
                "bbox": (
                    minx,
                    miny,
                    maxx - minx + 1,
                    maxy - miny + 1,
                ),  # (x, y, width, height)
                "centroid": (cx, cy),  # Center point of the blob
            }
        )

    return components


def dbscan(points, eps, min_pts):
    labels = [-1] * len(points)  # -1 means unclassified/noise
    cluster_id = 0

    def neighbors(idx):
        return [
            j
            for j, p in enumerate(points)
            if np.linalg.norm(np.array(p) - np.array(points[idx])) < eps
        ]

    # Main DBSCAN algorithm
    for i in range(len(points)):
        if labels[i] != -1:  # Already processed
            continue

        nbrs = neighbors(i)
        if len(nbrs) < min_pts:  # Not enough neighbors -> noise
            continue

        # Start a new cluster
        labels[i] = cluster_id
        queue = deque(nbrs)

        while queue:
            j = queue.popleft()
            if labels[j] == -1:  # Unclassified point
                labels[j] = cluster_id
            elif labels[j] != -1:  # Already in another cluster
                continue

            # Expand cluster if this point has enough neighbors
            nbrs2 = neighbors(j)
            if len(nbrs2) >= min_pts:
                queue.extend(nbrs2)

        cluster_id += 1

    # Group points by cluster and calculate centers
    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(points[idx])

    # Calculate center of each cluster
    centers = [tuple(np.mean(clusters[c], axis=0)) for c in sorted(clusters)]
    return centers


def smooth(series, win=widnow):
    if len(series) >= win:
        w = win if win % 2 == 1 else win + 1  # Ensure window is odd
        return savgol_filter(series, w, 3)  # 3rd order polynomial
    return series


def compute_motion(track, fps):
    arr = np.array(track, float)
    if len(arr) < 3:  # Need enough points for derivatives
        return (), (), (), ()

    x, y = arr[:, 0], arr[:, 1]
    x = smooth(x)  # Smooth x positions
    y = smooth(y)  # Smooth y positions

    dt = 1 / fps  # Time between frames

    # Calculate derivatives using numpy gradient
    vx = np.gradient(x, dt)  # Velocity in x direction (speed)
    vy = np.gradient(y, dt)  # Velocity in y direction
    ax = np.gradient(vx, dt)  # Acceleration in x direction
    ay = np.gradient(vy, dt)  # Acceleration in y direction
    jx = np.gradient(ax, dt)  # Jerk in x direction (rate of acceleration change)
    jy = np.gradient(ay, dt)  # Jerk in y direction
    sx = np.gradient(jx, dt)  # Jounce in x direction (rate of jerk change)
    sy = np.gradient(jy, dt)  # Jounce in y direction

    # For simplicity, we'll use x-direction only
    speed = vx
    acc = ax
    jerk = jx
    jounce = sx

    return speed, acc, jerk, jounce


# Main video processing loop
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Get video frames per second

# Read first frame
ok, frame_prev = cap.read()
if not ok:
    raise RuntimeError("Cannot read video")

frame_prev_gray = to_gray(frame_prev)
tracks = []  # Store motion tracks for each detected object

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Motion detection between current and previous frame
    gray = to_gray(frame)
    diff = abs_diff(gray, frame_prev_gray)  # Find differences
    motion = binarize(diff, threshold)  # Convert to binary mask

    # Clean up the motion mask
    mask = morph_open(motion, kernel)  # Remove noise

    # Dilate to connect nearby motion regions
    for _ in range(dilate):
        mask = cv2.dilate(mask, np.ones((kernel, kernel)))

    # Find motion blobs
    comps = connected_components(mask)

    # Filter small blobs and get their center points
    points = [c["centroid"] for c in comps if c["area"] >= blob_area]

    # Cluster nearby points (handle multiple detections of same object)
    centers = dbscan(points, eps, dbscan_min)

    # Update object tracks
    if not tracks:
        # First frame: create new tracks for each detected center
        tracks = [[c] for c in centers]
    else:
        # Match new detections with existing tracks
        last_pos = [t[-1] for t in tracks]  # Last known positions
        used = set()  # Track which new centers we've used

        # For each existing track, find closest new center
        for i, p in enumerate(last_pos):
            best = None
            best_dist = 1e9
            for j, c in enumerate(centers):
                if j in used:
                    continue
                d = np.linalg.norm(np.array(p) - np.array(c))
                if d < best_dist:
                    best_dist = d
                    best = j
            if best is not None:
                tracks[i].append(centers[best])  # Add to existing track
                used.add(best)

        # Create new tracks for unmatched centers
        for j, c in enumerate(centers):
            if j not in used:
                tracks.append([c])  # Start new track

    # Display results
    disp = frame.copy()

    # Draw current positions as RED circles (was yellow)
    for t in tracks:
        cv2.circle(disp, tuple(map(int, t[-1])), 5, (0, 0, 255), -1)

    # Draw bounding boxes around motion regions in RED (was green)
    for c in comps:
        if c["area"] >= blob_area:
            x, y, w, h = c["bbox"]
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Tracking", disp)
    frame_prev_gray = gray  # Update previous frame

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
# Analyze and plot motion characteristics
labels = ["Speed", "Acceleration", "Jerk", "Jounce"]

# Sort tracks by length (longest first) and take top 4
tracks_sorted = sorted(tracks, key=lambda t: -len(t))

# Create a single figure with subplots for all objects
plt.figure(figsize=(15, 10))
plt.suptitle("Motion Analysis for Top 4 Tracked Objects", fontsize=16)

# Plot each object in its own column
for idx, track in enumerate(tracks_sorted[:4]):
    spd, acc, jrk, jnc = compute_motion(track, fps)
    data = [spd, acc, jrk, jnc]

    # Create 4 subplots for each motion characteristic (rows) for this object (column)
    for k in range(4):
        plt.subplot(
            4, 4, idx + 1 + k * 4
        )  # 4 rows, 4 columns, arranged by characteristic
        plt.plot(data[k])
        plt.title(f"Object {idx+1} - {labels[k]}")
        plt.xlabel("Frame")
        plt.ylabel(labels[k])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
