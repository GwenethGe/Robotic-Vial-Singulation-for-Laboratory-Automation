"""
Grasp Candidate Generation and Scoring Module

Implements a traditional grasp synthesis pipeline:
1. Generate multiple grasp candidates (8-16 per object)
2. Score each candidate using:
   - Ferrari-Canny wrench space metric
   - Friction cone feasibility
   - Gripper aperture feasibility
   - IK reachability
3. Select and execute the highest-scoring grasp
"""
import numpy as np
from scipy.spatial.transform import Rotation
import mengine as m
from config import *
from force_grasping import contact_screw_3d, friction_cone_3d, is_force_closure

class GraspCandidate:
    """Represents a single grasp candidate"""
    def __init__(self, position, orientation, approach_angle, height_offset, grip_width):
        self.position = np.array(position)
        self.orientation = orientation  # quaternion
        self.approach_angle = approach_angle  # radians
        self.height_offset = height_offset  # meters
        self.grip_width = grip_width  # meters
        
        # Scores (filled during evaluation)
        self.ferrari_canny_score = 0.0
        self.friction_score = 0.0
        self.aperture_score = 0.0
        self.reachability_score = 0.0
        self.collision_score = 0.0
        self.total_score = 0.0
        
        self.ik_solution = None
        self.is_valid = False

def generate_grasp_candidates(vial_pos, vial_orient, vial_radius, vial_height, 
                              num_rotations=16, num_heights=5, num_approaches=3,
                              vial_idx=0):
    """
    Generate MANY grasp candidates for a vial.
    
    More candidates = more flexibility in finding collision-free paths.
    
    Args:
        vial_pos: Vial center position [x, y, z]
        vial_orient: Vial orientation quaternion
        vial_radius: Vial radius in meters
        vial_height: Vial height in meters
        num_rotations: Number of rotation angles around Z axis (default: 16)
        num_heights: Number of height offsets to try (default: 5)
        num_approaches: Number of approach angles (default: 3)
        vial_idx: Index of vial (for deterministic variation)
    
    Returns:
        List of GraspCandidate objects
        
    Total candidates = num_rotations * num_heights * num_approaches
    Default: 16 * 5 * 3 = 240 candidates
    """
    candidates = []
    
    # Gripper finger length (from gripper geometry)
    GRIPPER_FINGER_LENGTH = 0.050  # 5cm finger depth
    
    # Get vial axis in world frame
    R_vial = Rotation.from_quat(vial_orient)
    vial_z_axis = R_vial.apply([0, 0, 1])
    
    # Check if vial is mostly vertical or horizontal
    is_vertical = abs(vial_z_axis[2]) > 0.7
    
    # Detect large tubes (height > 50mm)
    is_large_tube = vial_height > 0.050
    
    # Height offsets: ADAPTIVE based on vial type and orientation
    if is_large_tube and is_vertical:
        # Large vertical tubes: grasp from TOP to avoid tilting
        # Grasp higher up (near top) to stay within gripper finger reach
        # If tube is 55mm tall and finger is 50mm, grasp at top 1/3
        # OPTIMIZED: More concentrated around optimal height (25% up)
        height_offsets = np.linspace(vial_height * 0.20, vial_height * 0.30, num_heights)
    elif is_large_tube and not is_vertical:
        # Large horizontal tubes: grasp higher than center to compensate for length
        # This prevents bottom from dragging when lifted
        # OPTIMIZED: More concentrated around optimal height (1-1.5cm above center)
        height_offsets = np.linspace(0.010, 0.015, num_heights)
    else:
        # Small/medium tubes: standard centered grasp
        # OPTIMIZED: More concentrated near center
        height_offsets = np.linspace(-0.008, 0.008, num_heights)
    
    # Rotation angles: full 360° coverage with more samples
    base_offset = (vial_idx * 7) % 45  # Unique offset per vial
    rotation_angles = np.linspace(np.radians(base_offset), 2*np.pi + np.radians(base_offset), 
                                   num_rotations, endpoint=False)
    
    # Approach angles: more options including tilted approaches
    approach_angles = np.linspace(0, np.radians(25), num_approaches)
    
    # Single grip width (we'll score based on this later)
    grip_width = vial_radius * 2 * 0.85  # 85% of diameter
    
    for rot_angle in rotation_angles:
        for height_off in height_offsets:
            for approach in approach_angles:
                
                # Calculate grasp position
                # CRITICAL FIX: For tilted vials, grasp at CENTER + vertical offset
                # NOT along vial axis (which causes position errors)
                if is_vertical:
                    # Top-down grasp for vertical vials
                    grasp_pos = vial_pos + np.array([0, 0, height_off])
                else:
                    # For horizontal/tilted vials: grasp at CENTER
                    # Apply VERTICAL offset only (not along vial axis)
                    # This ensures gripper descends straight down to vial center
                    grasp_pos = vial_pos + np.array([0, 0, height_off])
                
                # Calculate gripper orientation
                # ALWAYS use top-down grasp (most reliable for IK)
                # For horizontal vials, we adjust the grasp POSITION instead
                base_euler = [np.pi, 0, rot_angle]
                
                # Apply approach angle (tilt)
                if approach != 0:
                    base_euler[0] = np.pi - approach
                
                gripper_orient = Rotation.from_euler('xyz', base_euler).as_quat()
                
                candidate = GraspCandidate(
                    position=grasp_pos.copy(),
                    orientation=gripper_orient,
                    approach_angle=approach,
                    height_offset=height_off,
                    grip_width=grip_width
                )
                candidate.rotation_angle = rot_angle  # Store for scoring
                candidates.append(candidate)
    
    return candidates

def score_grasp_candidate(candidate, robot, vial, vial_mass, vial_radius, 
                          mu=VIAL_FRICTION, obstacles=None, all_vials=None, 
                          vial_idx=None, vial_pos=None):
    """
    Score a grasp candidate using multiple metrics.
    
    Metrics:
    1. Ferrari-Canny: Wrench space quality (from HW5)
    2. Friction: Friction cone feasibility
    3. Aperture: Gripper can achieve required width
    4. Reachability: IK solution exists
    5. Collision: No collision with obstacles
    6. Isolation: Distance to other vials (CRITICAL - penalize grabbing multiple)
    7. Geometry: Favor certain approach angles and heights
    
    Returns:
        Updated candidate with scores
    """
    # Weights - isolation is most important to avoid multi-grasp
    weights = {
        'ferrari_canny': 0.10,
        'friction': 0.05,
        'aperture': 0.05,
        'reachability': 0.20,  # Increased - IK feasibility is critical
        'collision': 0.15,     # Increased - collision avoidance is critical
        'isolation': 0.30,     # Avoid multi-grasp
        'geometry': 0.15       # Favor good approach angles
    }
    
    # 1. Reachability Score (IK) - Check MULTIPLE positions along approach path
    try:
        tcp_pos = candidate.position + np.array([0, 0, 0.05])
        
        # Check IK for grasp position
        ik_solution = robot.ik(robot.end_effector, tcp_pos, candidate.orientation,
                               max_iterations=100)
        if ik_solution is None:
            candidate.reachability_score = 0.0
            candidate.is_valid = False
            candidate.total_score = 0.0
            return candidate
        
        candidate.ik_solution = ik_solution
        
        # Also check IK for approach position (above grasp)
        approach_pos = tcp_pos + np.array([0, 0, 0.15])  # 15cm above
        approach_ik = robot.ik(robot.end_effector, approach_pos, candidate.orientation,
                               max_iterations=50)
        
        if approach_ik is not None:
            candidate.reachability_score = 1.0
        else:
            # Can reach grasp but not approach - partial score
            candidate.reachability_score = 0.5
            
    except:
        candidate.reachability_score = 0.0
        candidate.is_valid = False
        candidate.total_score = 0.0
        return candidate
    
    # 2. Aperture Score
    max_aperture = 0.08  # Panda gripper max opening ~8cm
    if candidate.grip_width <= max_aperture:
        candidate.aperture_score = 1.0 - (candidate.grip_width / max_aperture) * 0.3
    else:
        candidate.aperture_score = 0.0
        candidate.is_valid = False
        candidate.total_score = 0.0
        return candidate
    
    # 3. Ferrari-Canny Score (Force Closure Quality)
    R_gripper = Rotation.from_quat(candidate.orientation)
    gripper_y = R_gripper.apply([0, 1, 0])  # Finger closing direction
    
    # Contact points on opposite sides of vial
    contact1 = candidate.position + gripper_y * vial_radius
    contact2 = candidate.position - gripper_y * vial_radius
    
    # Contact normals point inward
    normal1 = -gripper_y
    normal2 = gripper_y
    
    contact_points = np.array([contact1, contact2])
    contact_normals = np.array([normal1, normal2])
    
    try:
        W, primitives = friction_cone_3d(contact_points, contact_normals, mu, n_fc=8)
        is_fc, z_max = is_force_closure(W)
        
        if is_fc:
            candidate.ferrari_canny_score = min(z_max / 3.0, 1.0)
        else:
            candidate.ferrari_canny_score = 0.1
    except:
        candidate.ferrari_canny_score = 0.3
    
    # 4. Friction Score
    candidate.friction_score = min(mu / 5.0, 1.0)
    
    # 5. Collision Score (check height, position, AND box collision)
    # Use vial position as reference (not hardcoded TABLE_HEIGHT)
    # ADAPTIVE height tolerance based on vial size and orientation
    if vial_pos is not None:
        height_relative_to_vial = candidate.position[2] - vial_pos[2]  # Signed offset
        
        # Get vial orientation to determine if it's vertical
        from robot_control import get_orientation, get_vial_axis
        vial_orient = get_orientation(vial)
        vial_axis = get_vial_axis(vial_orient)
        is_vertical = abs(vial_axis[2]) > 0.7
        
        # ADAPTIVE tolerance based on vial type and orientation
        # For vertical vials: allow grasping higher (near top)
        # For horizontal vials: stay near center
        if is_vertical:
            # Vertical vials: prefer grasping from TOP (positive offset)
            # Allow up to half vial height above center
            # Large vial: 55mm tall → allow up to +27mm
            vial_height = candidate.height_offset  # This is already the intended offset
            if -0.01 < height_relative_to_vial < 0.035:  # -1cm to +3.5cm
                candidate.collision_score = 1.0  # Good height range
            elif height_relative_to_vial < -0.02:  # Too low (below center)
                candidate.collision_score = 0.3  # Strong penalty
            else:  # Slightly too high but acceptable
                candidate.collision_score = 0.7
        else:
            # Horizontal/tilted vials: stay near center
            # Tighter tolerance: ±2cm
            if abs(height_relative_to_vial) < 0.02:
                candidate.collision_score = 1.0
            elif abs(height_relative_to_vial) < 0.04:
                candidate.collision_score = 0.7
            else:
                candidate.collision_score = 0.4
    else:
        # Fallback: use absolute height check
        height_above_table = candidate.position[2] - TABLE_HEIGHT
        
        if height_above_table < 0.02:
            candidate.collision_score = 0.2
        elif height_above_table > 0.15:
            candidate.collision_score = 0.6
        else:
            candidate.collision_score = 1.0
    
    # Check collision with box walls AND ensure grasp is reachable
    box_center = np.array(BOX_POS)
    box_half = np.array(BOX_SIZE) / 2
    box_wall_top = TABLE_HEIGHT + BOX_SIZE[2]
    
    # Gripper/arm dimensions
    GRIPPER_WIDTH = 0.08   # 8cm gripper width when open
    ARM_CLEARANCE = 0.08   # 8cm clearance for entire arm (reduced from 10cm for better edge access)
    
    # Get gripper orientation
    R_gripper = Rotation.from_quat(candidate.orientation)
    gripper_z = R_gripper.apply([0, 0, 1])  # Gripper pointing direction (usually down)
    gripper_y = R_gripper.apply([0, 1, 0])  # Finger closing direction
    
    # Check 1: Grasp point must be well inside box (not at edge)
    # Vial center might be at edge, but gripper needs space
    grasp_margin_x = abs(candidate.position[0] - box_center[0]) + GRIPPER_WIDTH/2
    grasp_margin_y = abs(candidate.position[1] - box_center[1]) + GRIPPER_WIDTH/2
    
    if grasp_margin_x > box_half[0] or grasp_margin_y > box_half[1]:
        # Grasp point too close to edge - gripper fingers would be outside
        # INVALIDATE this grasp completely
        candidate.is_valid = False
        candidate.collision_score = 0.0
        candidate.total_score = 0.0
        return candidate
    
    # Check 2: Arm collision along approach path
    for dist in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:  # Points from grasp to 25cm above
        check_pos = candidate.position - gripper_z * dist
        
        # Check if arm is outside box bounds
        outside_x = abs(check_pos[0] - box_center[0]) > box_half[0] - ARM_CLEARANCE
        outside_y = abs(check_pos[1] - box_center[1]) > box_half[1] - ARM_CLEARANCE
        below_wall = check_pos[2] < box_wall_top + 0.02
        
        if (outside_x or outside_y) and below_wall:
            # Arm would collide with box walls - INVALIDATE
            candidate.is_valid = False
            candidate.collision_score = 0.0
            candidate.total_score = 0.0
            return candidate
    
    # Check if grasp point is near box edge (approach collision risk)
    dist_to_box_edge_x = abs(candidate.position[0] - box_center[0]) - box_half[0]
    dist_to_box_edge_y = abs(candidate.position[1] - box_center[1]) - box_half[1]
    
    # Near edge = within 8cm of wall
    near_edge_x = -0.08 < dist_to_box_edge_x < 0.08
    near_edge_y = -0.08 < dist_to_box_edge_y < 0.08
    
    if near_edge_x or near_edge_y:
        # Near box edge - penalize based on distance
        edge_dist = min(abs(dist_to_box_edge_x), abs(dist_to_box_edge_y))
        penalty = 0.5 + 0.5 * (edge_dist / 0.08)  # 0.5 to 1.0
        candidate.collision_score *= penalty
    
    # 6. Isolation Score (CRITICAL - avoid grabbing multiple vials)
    # Use demo's sophisticated isolation scoring system
    candidate.isolation_score = 1.0  # Default: perfect isolation
    candidate.nearest_vial_dist = float('inf')  # Store for debugging
    candidate.arm_collision_score = 1.0  # New: arm collision penalty
    
    if all_vials is not None and vial_idx is not None:
        from robot_control import get_position
        from task_logic import compute_isolation_score
        
        # Use demo's multi-factor isolation score (0-100)
        # This considers: XY density, 3D separation, boundary, occlusion, orientation
        demo_isolation_score = compute_isolation_score(vial_idx, all_vials)
        
        # Normalize to 0-1 for grasp scoring
        candidate.isolation_score = demo_isolation_score / 100.0
        
        # Gripper physical dimensions
        GRIPPER_FINGER_WIDTH = 0.02  # 2cm per finger
        GRIPPER_OPEN_WIDTH = 0.08    # 8cm total opening
        GRIPPER_DEPTH = 0.04         # 4cm finger depth
        
        # ARM dimensions for collision checking
        ARM_RADIUS = 0.05  # 5cm radius cylinder for arm
        ARM_LENGTH = 0.30  # Check 30cm of arm above grasp point
        
        # Get gripper orientation
        R_gripper = Rotation.from_quat(candidate.orientation)
        gripper_y = R_gripper.apply([0, 1, 0])  # Finger direction
        gripper_x = R_gripper.apply([1, 0, 0])  # Forward direction
        gripper_z = R_gripper.apply([0, 0, 1])  # Arm direction (up from grasp)
        
        # Finger tip positions (when gripper is open)
        finger1_pos = candidate.position + gripper_y * (GRIPPER_OPEN_WIDTH/2)
        finger2_pos = candidate.position - gripper_y * (GRIPPER_OPEN_WIDTH/2)
        
        min_dist_to_other = float('inf')
        closest_vial_idx = -1
        arm_collision_detected = False
        
        for i, other_vial in enumerate(all_vials):
            if i == vial_idx:
                continue
            
            other_pos = get_position(other_vial)
            
            # Check ARM collision with this tube
            # Check points along the arm (from grasp to 30cm above)
            for arm_dist in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
                arm_point = candidate.position - gripper_z * arm_dist  # Arm goes opposite to gripper Z
                dist_to_tube = np.linalg.norm(arm_point - other_pos)
                
                # If arm point is within tube radius + arm radius, collision!
                if dist_to_tube < ARM_RADIUS + vial_radius:
                    arm_collision_detected = True
                    break
            
            # Check multiple collision points for gripper:
            # 1. Distance from grasp center to other vial
            dist_center = np.linalg.norm(candidate.position - other_pos)
            
            # 2. Distance from each finger to other vial
            dist_finger1 = np.linalg.norm(finger1_pos - other_pos)
            dist_finger2 = np.linalg.norm(finger2_pos - other_pos)
            
            # 3. Check if other vial is in the gripper "sweep zone"
            to_other = other_pos - candidate.position
            lateral_dist = abs(np.dot(to_other, gripper_y))
            forward_dist = abs(np.dot(to_other, gripper_x))
            vertical_dist = abs(to_other[2])
            
            # Minimum distance considering all factors
            # If vial is between the fingers (lateral < opening/2) and close vertically
            if lateral_dist < GRIPPER_OPEN_WIDTH/2 + 0.02 and vertical_dist < 0.05:
                # This vial could be grabbed!
                effective_dist = forward_dist  # How far in front/behind
            else:
                effective_dist = min(dist_center, dist_finger1, dist_finger2)
            
            if effective_dist < min_dist_to_other:
                min_dist_to_other = effective_dist
                closest_vial_idx = i
        
        candidate.nearest_vial_dist = min_dist_to_other
        
        # COMBINE demo's global isolation score with gripper-specific checks
        # Demo score (0-1): considers spatial layout, boundaries, occlusion
        # Gripper score (0-1): considers immediate multi-grasp risk
        
        # Gripper-specific isolation using exponential decay
        k = 30  # Steepness factor
        gripper_isolation = 1.0 - np.exp(-k * min_dist_to_other)
        
        # Additional penalty for very close vials
        if min_dist_to_other < 0.025:
            gripper_isolation *= 0.3  # Heavy penalty
        elif min_dist_to_other < 0.035:
            gripper_isolation *= 0.6  # Moderate penalty
        
        # Combine: use minimum (most conservative)
        # This ensures BOTH global and local isolation are good
        candidate.isolation_score = min(candidate.isolation_score, gripper_isolation)
        
        # ARM COLLISION PENALTY
        if arm_collision_detected:
            candidate.arm_collision_score = 0.1  # Severe penalty - arm would hit a tube
            candidate.collision_score *= 0.2    # Also affect overall collision score
    
    # 7. Geometry Score - OPTIMIZED FOR LARGE TUBES
    # Prefer smaller approach angles (more vertical = safer)
    approach_penalty = (candidate.approach_angle / np.radians(30)) ** 2  # Quadratic penalty
    
    # INCREASED height penalty weight - CRITICAL for large tubes
    # For large tubes, grasping at wrong height causes tilting/dropping
    # Increased exponent from 2.0 to 2.5 for even stronger penalty
    height_penalty = (abs(candidate.height_offset) / 0.01) ** 2.5  # Stronger penalty (was 2.0)
    
    # Add random small noise to break ties
    noise = np.random.uniform(0, 0.05)
    
    # REMOVED rot_preference: Z-axis rotation doesn't affect reachability
    # All rotation angles (0°, 90°, 180°, 270°) are equally valid for IK
    # This gives all rotations equal opportunity, especially important for edge vials
    
    # INCREASED height penalty weight from 0.5 to 0.6 (CRITICAL for large tubes)
    # Reduced approach penalty from 0.3 to 0.2 to prioritize height over approach angle
    candidate.geometry_score = max(0, 1.0 - 0.2*approach_penalty - 0.6*height_penalty + noise)
    
    # Calculate total weighted score
    candidate.total_score = (
        weights['ferrari_canny'] * candidate.ferrari_canny_score +
        weights['friction'] * candidate.friction_score +
        weights['aperture'] * candidate.aperture_score +
        weights['reachability'] * candidate.reachability_score +
        weights['collision'] * candidate.collision_score +
        weights['isolation'] * candidate.isolation_score +
        weights['geometry'] * candidate.geometry_score
    )
    
    candidate.is_valid = True
    return candidate

def select_best_grasp(candidates, robot, vial, vial_mass, vial_radius, 
                      mu=VIAL_FRICTION, top_k=5, verbose=True,
                      all_vials=None, vial_idx=None):
    """
    Score all candidates and select the best one.
    
    Args:
        candidates: List of GraspCandidate objects
        robot: Robot object for IK checking
        vial: Vial object
        vial_mass: Vial mass in kg
        vial_radius: Vial radius in meters
        mu: Friction coefficient
        top_k: Number of top candidates to report
        verbose: Print scoring details
        all_vials: List of all vials (for isolation scoring)
        vial_idx: Index of target vial
    
    Returns:
        Best GraspCandidate or None if no valid grasp found
    """
    from robot_control import get_position
    vial_pos = get_position(vial)
    
    scored_candidates = []
    
    for i, candidate in enumerate(candidates):
        scored = score_grasp_candidate(
            candidate, robot, vial, vial_mass, vial_radius, mu,
            all_vials=all_vials, vial_idx=vial_idx, vial_pos=vial_pos
        )
        if scored.is_valid:
            scored_candidates.append(scored)
    
    if not scored_candidates:
        if verbose:
            print("   No valid grasp candidates found!")
        return None
    
    # Sort by total score (descending)
    scored_candidates.sort(key=lambda c: c.total_score, reverse=True)
    
    if verbose:
        print(f"   Grasp Candidate Scoring ({len(scored_candidates)} valid):")
        for i, c in enumerate(scored_candidates[:top_k]):
            iso_str = f"Iso={c.isolation_score:.2f}" if hasattr(c, 'isolation_score') else ""
            geo_str = f"Geo={c.geometry_score:.2f}" if hasattr(c, 'geometry_score') else ""
            print(f"     #{i+1}: Score={c.total_score:.3f} "
                  f"[FC={c.ferrari_canny_score:.2f}, "
                  f"{iso_str}, {geo_str}, "
                  f"θ={np.degrees(c.approach_angle):.0f}°]")
    
    return scored_candidates[0]

def check_grasp_isolation(vial_pos, all_vials, vial_idx, min_distance=0.05):
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
    from robot_control import get_position
    
    nearest_dist = float('inf')
    nearest_idx = -1
    crowded_count = 0  # How many vials are too close
    
    DANGER_ZONE = 0.04  # 4cm - gripper could grab both
    WARNING_ZONE = 0.08  # 8cm - needs careful approach
    
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


def find_most_isolated_vial(all_vials, vial_picked, vial_types, target_type=None):
    """
    Find the vial that is most isolated from others.
    This helps avoid multi-grasp situations.
    
    Args:
        all_vials: List of all vial objects
        vial_picked: List of booleans indicating if each vial was picked
        vial_types: List of vial types
        target_type: If specified, only consider vials of this type
    
    Returns:
        (best_idx, isolation_score) or (None, 0) if no valid vial found
    """
    from robot_control import get_position
    
    best_idx = None
    best_isolation = 0
    
    for i, vial in enumerate(all_vials):
        if vial_picked[i]:
            continue
        
        if target_type is not None and vial_types[i] != target_type:
            continue
        
        vial_pos = get_position(vial)
        
        # Calculate isolation score (minimum distance to any other unpicked vial)
        min_dist = float('inf')
        for j, other_vial in enumerate(all_vials):
            if i == j or vial_picked[j]:
                continue
            other_pos = get_position(other_vial)
            dist = np.linalg.norm(vial_pos - other_pos)
            min_dist = min(min_dist, dist)
        
        if min_dist > best_isolation:
            best_isolation = min_dist
            best_idx = i
    
    return best_idx, best_isolation

