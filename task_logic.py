"""
Task planning and logic functions (Simple version)
"""
import numpy as np
import pybullet as p
from config import *
from robot_control import get_position, get_orientation, get_vial_axis

# Global variable to store debug text IDs for updating
_isolation_score_text_ids = []

def check_grasp_isolation(vial_pos, all_vials, vial_idx, min_distance=0.04):
    """
    Check if a vial is isolated enough for safe grasping.
    Prevents grabbing multiple vials at once.
    
    Args:
        vial_pos: Position of target vial
        all_vials: List of all vial objects
        vial_idx: Index of target vial
        min_distance: Minimum distance to other vials (meters) - default 5cm
    
    Returns:
        (is_isolated, nearest_distance, nearest_idx, crowded_count)
    """
    nearest_dist = float('inf')
    nearest_idx = -1
    crowded_count = 0  # How many vials are too close
    
    DANGER_ZONE = 0.04  # 4cm - gripper could grab both
    
    for i, other_vial in enumerate(all_vials):
        if i == vial_idx:
            continue
        
        other_pos = get_position(other_vial)
        dist = np.linalg.norm(vial_pos - other_pos)
        
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_idx = i
        
        if dist < DANGER_ZONE:
            crowded_count += 1
    
    # Stricter isolation check
    is_isolated = nearest_dist > min_distance and crowded_count == 0
    
    return is_isolated, nearest_dist, nearest_idx, crowded_count

def compute_isolation_score(vial_idx, all_vials):
    """
    Compute isolation score based on 3D spatial density and accessibility.
    
    Considers:
    - XY plane density (most important for top-down grasp)
    - 3D spatial separation
    - Box boundary proximity (penalty for wall closeness)
    - Occlusion from above (not just height)
    - Nearest neighbor distance
    
    Args:
        vial_idx: Index of target vial
        all_vials: List of all vial objects
    
    Returns:
        isolation_score: Float (0-100, higher = more isolated + accessible)
    """
    from config import BOX_POS, BOX_SIZE
    
    vial_pos = get_position(all_vials[vial_idx])
    
    # Collect both 3D and XY distances
    distances_3d = []
    distances_xy = []
    occlusion_data = []  # (xy_dist, height_diff) for occlusion analysis
    
    for i, other_vial in enumerate(all_vials):
        if i == vial_idx:
            continue
        other_pos = get_position(other_vial)
        
        # 3D distance for true spatial separation
        dist_3d = np.linalg.norm(vial_pos - other_pos)
        distances_3d.append(dist_3d)
        
        # XY distance for horizontal crowding
        dist_xy = np.linalg.norm(vial_pos[:2] - other_pos[:2])
        distances_xy.append(dist_xy)
        
        # Store for occlusion analysis
        height_diff = vial_pos[2] - other_pos[2]
        occlusion_data.append((dist_xy, height_diff))
    
    if not distances_3d:
        return 100.0
    
    # Component 1: XY Plane Density (0-50 points)
    # Grasping primarily depends on XY clearance
    count_xy_0_5cm = sum(1 for d in distances_xy if d < 0.05)
    count_xy_5_10cm = sum(1 for d in distances_xy if 0.05 <= d < 0.10)
    count_xy_10_15cm = sum(1 for d in distances_xy if 0.10 <= d < 0.15)
    
    # Inner XY zone (0-5cm): 0=25pts, 1=12pts, 2=3pts, 3+=0pts
    if count_xy_0_5cm == 0:
        xy_inner = 25.0
    elif count_xy_0_5cm == 1:
        xy_inner = 12.0
    elif count_xy_0_5cm == 2:
        xy_inner = 3.0
    else:
        xy_inner = 0.0
    
    # Middle XY zone (5-10cm): 0=15pts, 1-2=10pts, 3+=5pts
    if count_xy_5_10cm == 0:
        xy_middle = 15.0
    elif count_xy_5_10cm <= 2:
        xy_middle = 10.0
    else:
        xy_middle = 5.0
    
    # Outer XY zone (10-15cm): 0-3=10pts, 4-6=7pts, 7+=3pts
    if count_xy_10_15cm <= 3:
        xy_outer = 10.0
    elif count_xy_10_15cm <= 6:
        xy_outer = 7.0
    else:
        xy_outer = 3.0
    
    xy_score = xy_inner + xy_middle + xy_outer
    
    # Component 2: 3D Spatial Separation (0-25 points)
    # True 3D isolation matters for overall accessibility
    count_3d_0_6cm = sum(1 for d in distances_3d if d < 0.06)
    count_3d_6_12cm = sum(1 for d in distances_3d if 0.06 <= d < 0.12)
    
    # Close in 3D space (0-6cm): 0=15pts, 1=8pts, 2+=0pts
    if count_3d_0_6cm == 0:
        spatial_close = 15.0
    elif count_3d_0_6cm == 1:
        spatial_close = 8.0
    else:
        spatial_close = 0.0
    
    # Medium 3D distance (6-12cm): 0-2=10pts, 3-5=5pts, 6+=0pts
    if count_3d_6_12cm <= 2:
        spatial_medium = 10.0
    elif count_3d_6_12cm <= 5:
        spatial_medium = 5.0
    else:
        spatial_medium = 0.0
    
    spatial_score = spatial_close + spatial_medium
    
    # Component 3: Box Boundary Penalty (0-35 points) - INCREASED WEIGHT
    # Vials near walls are VERY hard to grasp (gripper collision risk)
    # Gripper upper part is ~10cm wide, needs clearance
    box_x_min = BOX_POS[0] - BOX_SIZE[0]/2
    box_x_max = BOX_POS[0] + BOX_SIZE[0]/2
    box_y_min = BOX_POS[1] - BOX_SIZE[1]/2
    box_y_max = BOX_POS[1] + BOX_SIZE[1]/2
    
    # Distance to nearest wall
    dist_to_x_walls = min(abs(vial_pos[0] - box_x_min), abs(vial_pos[0] - box_x_max))
    dist_to_y_walls = min(abs(vial_pos[1] - box_y_min), abs(vial_pos[1] - box_y_max))
    dist_to_wall = min(dist_to_x_walls, dist_to_y_walls)
    
    # STRICTER penalty for being close to walls
    # Gripper needs 10-12cm clearance for tilted approach
    if dist_to_wall > 0.15:  # 15cm+ from wall (very safe)
        boundary_score = 35.0
    elif dist_to_wall > 0.12:  # 12-15cm (safe)
        boundary_score = 28.0
    elif dist_to_wall > 0.10:  # 10-12cm (borderline safe)
        boundary_score = 20.0
    elif dist_to_wall > 0.08:  # 8-10cm (risky, needs careful approach)
        boundary_score = 10.0
    elif dist_to_wall > 0.06:  # 6-8cm (very risky)
        boundary_score = 3.0
    else:  # < 6cm (extremely dangerous)
        boundary_score = 0.0
    
    # Store corner penalty separately (will subtract from total at end)
    corner_penalty = 0.0
    if dist_to_x_walls < 0.12 and dist_to_y_walls < 0.12:  # Both walls within 12cm
        # Severity based on how close to true corner
        corner_closeness = max(dist_to_x_walls, dist_to_y_walls)
        if corner_closeness < 0.08:
            corner_penalty = 40.0  # Very close to corner - near impossible
        elif corner_closeness < 0.10:
            corner_penalty = 25.0  # Corner proximity - very hard
        else:
            corner_penalty = 15.0  # Near corner - hard
    
    # Component 4: Vertical Occlusion Check (0-8 points)
    # Check if vial is actually occluded from above (not just lower)
    # A vial is occluded if there's another vial directly above it (close in XY, higher in Z)
    occluding_vials = 0
    for xy_dist, height_diff in occlusion_data:
        # Another vial is above (height_diff < 0) and close in XY
        if height_diff < -0.01 and xy_dist < 0.06:  # 6cm XY radius, 1cm+ above
            occluding_vials += 1
    
    # Penalize occluded vials
    if occluding_vials == 0:
        occlusion_score = 8.0  # Clear from above
    elif occluding_vials == 1:
        occlusion_score = 4.0  # Partially occluded
    else:
        occlusion_score = 0.0  # Heavily occluded
    
    # Component 5: Nearest Neighbor Bonus (0-10 points)
    nearest_3d = min(distances_3d)
    if nearest_3d > 0.12:  # 12cm in 3D
        nearest_bonus = 10.0
    elif nearest_3d > 0.08:
        nearest_bonus = 5.0
    elif nearest_3d > 0.05:
        nearest_bonus = 2.0
    else:
        nearest_bonus = 0.0
    
    # Component 6: Vial Orientation Bonus (0-12 points)
    # Prefer vials that are already upright (easier to place vertically in tray)
    vial_orient = get_orientation(all_vials[vial_idx])
    vial_axis = get_vial_axis(vial_orient)
    vertical = np.array([0, 0, 1])
    tilt_angle = np.arccos(np.clip(abs(np.dot(vial_axis, vertical)), 0, 1))
    tilt_degrees = np.degrees(tilt_angle)
    
    # Upright vials get bonus, highly tilted get penalty
    if tilt_degrees < 15:  # Nearly upright
        orientation_bonus = 12.0
    elif tilt_degrees < 30:  # Slightly tilted
        orientation_bonus = 8.0
    elif tilt_degrees < 45:  # Moderately tilted
        orientation_bonus = 4.0
    elif tilt_degrees < 60:  # Quite tilted
        orientation_bonus = 0.0
    else:  # Very tilted or horizontal - penalty
        orientation_bonus = -8.0
    
    # Calculate base score
    base_score = xy_score + spatial_score + boundary_score + occlusion_score + nearest_bonus + orientation_bonus
    
    # Apply multiplicative border penalty for very close vials
    # This ensures border vials CANNOT get high scores even if isolated
    border_multiplier = 1.0
    if dist_to_wall < 0.06:  # < 6cm: extremely hard
        border_multiplier = 0.25  # Reduce score to 25%
    elif dist_to_wall < 0.08:  # 6-8cm: very hard
        border_multiplier = 0.50  # Reduce score to 50%
    elif dist_to_wall < 0.10:  # 8-10cm: hard
        border_multiplier = 0.70  # Reduce score to 70%
    elif dist_to_wall < 0.12:  # 10-12cm: moderate
        border_multiplier = 0.85  # Reduce score to 85%
    
    # Apply border multiplier
    total_score = base_score * border_multiplier
    
    # Subtract corner penalty
    total_score = max(0.0, total_score - corner_penalty)
    
    return min(100.0, total_score)

def find_most_isolated_vial(all_vials, vial_picked, target_type=None):
    """
    Find the most isolated vial (highest isolation score).
    
    Args:
        all_vials: List of all vial objects
        vial_picked: Boolean array indicating which vials are already picked
        target_type: Optional - filter by vial type ('small', 'medium', 'large')
    
    Returns:
        best_idx: Index of most isolated vial, or None if no valid vial
    """
    best_idx = None
    best_score = -1
    
    for i in range(len(all_vials)):
        if vial_picked[i]:
            continue
        
        # TODO: Add type filtering if target_type is specified
        
        score = compute_isolation_score(i, all_vials)
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    return best_idx

def compute_target_slot_position(vtype, row, col):
    """Compute target position for a tray slot
    
    For trays with thick inner walls (like large tray), we need to account for:
    - outer_wall_width: 0.012m
    - inner_wall_width: config-specific (0.012m or 0.030m)
    
    Slot (i,j) position:
    x = tray_center - frame_half_x + outer_wall + (i + 0.5) * slot + i * inner_wall
    """
    tray_config = TRAY_CONFIGS[vtype]
    slot_size = tray_config['slot_size']
    tray_pos = tray_config['position']
    rows = tray_config['rows']
    cols = tray_config['cols']
    wall_height = tray_config.get('wall_height', TRAY_WALL_HEIGHT)
    
    # Wall widths
    outer_wall_width = 0.012
    inner_wall_width = tray_config.get('inner_wall_width', 0.012)
    
    # Total frame size
    frame_width_x = outer_wall_width * 2 + cols * slot_size + (cols - 1) * inner_wall_width
    frame_width_y = outer_wall_width * 2 + rows * slot_size + (rows - 1) * inner_wall_width
    frame_half_x = frame_width_x / 2
    frame_half_y = frame_width_y / 2
    
    # Slot center position
    slot_x = tray_pos[0] - frame_half_x + outer_wall_width + (col + 0.5) * slot_size + col * inner_wall_width
    slot_y = tray_pos[1] - frame_half_y + outer_wall_width + (row + 0.5) * slot_size + row * inner_wall_width
    slot_z = tray_pos[2] + wall_height + VIAL_SPECS[vtype]['height']/2 + 0.006
    
    return [slot_x, slot_y, slot_z]

def check_placement_success(vial, target_pos):
    """Check if vial was successfully placed"""
    final_pos = get_position(vial)
    dist = np.linalg.norm(final_pos[:2] - np.array(target_pos[:2]))
    return dist < PLACEMENT_TOLERANCE

def is_tray_full(vtype, placed_count):
    """Check if a tray is full"""
    tray_config = TRAY_CONFIGS[vtype]
    max_slots = tray_config['rows'] * tray_config['cols']
    return placed_count >= max_slots

def visualize_isolation_scores(all_vials, vial_skip_counts, label=""):
    """
    Visualize isolation scores above each vial in 3D.
    
    Args:
        all_vials: List of all vial objects
        vial_skip_counts: List of skip counts for each vial
        label: Additional label (e.g., "After push")
    """
    global _isolation_score_text_ids
    
    # Remove old text
    for text_id in _isolation_score_text_ids:
        try:
            p.removeUserDebugItem(text_id)
        except:
            pass
    _isolation_score_text_ids = []
    
    # Add new text for each vial
    for vial_idx, vial in enumerate(all_vials):
        vial_pos = get_position(vial)
        
        # Compute isolation score for this vial
        score = compute_isolation_score(vial_idx, all_vials)
        
        # Apply skip penalty
        skip_count = vial_skip_counts[vial_idx] if vial_idx < len(vial_skip_counts) else 0
        if skip_count > 0:
            score -= skip_count * 10
        
        # Choose color based on score
        if score > 80:
            color = [0, 1, 0]  # Green - excellent
        elif score > 60:
            color = [0.5, 1, 0]  # Yellow-green - good
        elif score > 40:
            color = [1, 1, 0]  # Yellow - okay
        elif score > 20:
            color = [1, 0.5, 0]  # Orange - poor
        else:
            color = [1, 0, 0]  # Red - very poor
        
        # Position text above vial
        text_pos = [vial_pos[0], vial_pos[1], vial_pos[2] + 0.08]
        
        # Create text with score
        skip_str = f" (x{skip_count})" if skip_count > 0 else ""
        text = f"{score:.0f}{skip_str}"
        
        text_id = p.addUserDebugText(
            text,
            text_pos,
            textColorRGB=color,
            textSize=1.2,
            lifeTime=0  # Permanent until manually removed
        )
        _isolation_score_text_ids.append(text_id)

