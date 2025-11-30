import math

def _angle_between(p1, p2):
    """Return angle (radians) of the vector p2 - p1."""
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

def _curvature_estimate(waypoints, closest_waypoints, lookahead=3):
    """
    Estimate the curvature ahead using waypoints.
    We use a simple angle difference between segments spaced by `lookahead`.
    waypoints: list of (x,y)
    closest_waypoints: pair of indices [i, i+1]
    """
    i = closest_waypoints[1]  # index of the next waypoint
    n = len(waypoints)
    # defensive indexing
    prev_idx = (i - 1) % n
    next_idx = (i + lookahead) % n

    ang1 = _angle_between(waypoints[prev_idx], waypoints[i])
    ang2 = _angle_between(waypoints[i], waypoints[next_idx])
    # minimal angular difference
    dtheta = abs((ang2 - ang1 + math.pi) % (2 * math.pi) - math.pi)
    return dtheta  # larger => sharper curve

def reward_function(params):
    """
    Curvature-aware time-trial reward.

    Combines:
      - center-line proximity,
      - speed vs a curvature-dependent target speed,
      - progress per step (small),
      - steering smoothness penalty,
      - big penalty for crashes / off-track.
    """

    # read commonly used params
    track_width = params.get('track_width', 1.0)
    distance_from_center = params.get('distance_from_center', 0.0)
    progress = params.get('progress', 0.0)
    speed = params.get('speed', 0.0)
    steering = params.get('steering_angle', 0.0)  # degrees
    is_crashed = params.get('is_crashed', False)
    is_offtrack = params.get('is_offtrack', False)
    waypoints = params.get('waypoints', [])
    closest_waypoints = params.get('closest_waypoints', [0, 1])
    steps = params.get('steps', 0)

    # immediate failure penalties
    if is_crashed or is_offtrack:
        return float(1e-3)

    # --- center-line reward (markers)
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width
    if distance_from_center <= marker_1:
        center_reward = 1.0
    elif distance_from_center <= marker_2:
        center_reward = 0.5
    elif distance_from_center <= marker_3:
        center_reward = 0.1
    else:
        center_reward = 1e-3

    # --- curvature-based target speed
    # Estimate curvature from waypoints; if not available, fallback to a simple heuristic
    try:
        curvature = _curvature_estimate(waypoints, closest_waypoints, lookahead=3)
    except Exception:
        curvature = 0.0

    # Map curvature (radians) to a target speed in [v_min, v_max]
    V_MIN = 1.0
    V_MAX = 2.0
    # curvature in [0, pi]; higher curvature -> lower target speed
    # We set target speed = V_MAX when curvature small, and closer to V_MIN when curvature is large
    k = 1.0  # curvature sensitivity; tune if needed
    target_speed = V_MIN + (V_MAX - V_MIN) * math.exp(-k * curvature)

    # speed reward: encourage speed closer to target (use a smooth function)
    # normalized to [0,1]
    speed_ratio = max(0.0, min(1.0, speed / (V_MAX + 1e-6)))
    # Prefer speed near target: reward = exp(- (speed - target)^2 / sigma)
    sigma = 0.5
    speed_reward = math.exp(-((speed - target_speed) ** 2) / (2 * sigma * sigma))

    # --- steering smoothness penalty
    ABS_STEERING_THRESHOLD = 15.0  # deg; tune if necessary
    steering_penalty = 1.0
    abs_steering = abs(steering)
    if abs_steering > ABS_STEERING_THRESHOLD:
        steering_penalty = 0.8  # mild penalty
    # stronger penalty for extreme steering
    if abs_steering > 25.0:
        steering_penalty *= 0.5

    # --- progress bonus (per-step)
    # Most reward shaping should be local; give a small reward proportional to incremental progress
    # Note: params['progress'] is cumulative percentage; we can't compute delta easily here,
    # but we can use steps and progress as a proxy (encourage increasing progress per step).
    progress_reward = progress / 100.0 / max(steps, 1)

    # --- combine terms (weights chosen for balanced behavior)
    w_center = 0.45
    w_speed = 0.35
    w_progress = 0.15
    w_steer = 0.10  # this is a multiplicative factor via steering_penalty

    reward = (
        w_center * center_reward +
        w_speed * speed_reward +
        w_progress * progress_reward
    )

    # apply steering penalty multiplicatively
    reward = reward * steering_penalty

    # Small bonus if agent is very near the center and fast on a straight
    if distance_from_center <= marker_1 and curvature < 0.05:
        reward += 0.1 * min(1.0, speed_ratio)

    # normalize and ensure positive
    reward = max(1e-3, float(reward))

    return reward
