"""
Grasp analysis and visualization module

Analyzes vial geometry and orientation to determine optimal grasp configuration:
- Best grasp position (height along vial)
- Optimal gripper orientation
- Approach angle recommendations
- Obstacle clearance
"""
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from robot_control import get_position, get_orientation, get_vial_axis


def analyze_vial_orientation(vial):
    """
    Analyze vial orientation and geometry.
    
    Returns:
        vial_pos: Vial center position
        vial_axis: Principal axis (cylinder axis)
        tilt_angle: Angle from vertical (radians)
        is_upright: True if vial is mostly vertical
        rotation_quat: Vial orientation quaternion
    """
    vial_pos = get_position(vial)
    vial_orient = get_orientation(vial)
    vial_axis = get_vial_axis(vial_orient)
    
    # Calculate tilt from vertical
    vertical = np.array([0, 0, 1])
    tilt_angle = np.arccos(np.clip(np.dot(vial_axis, vertical), -1.0, 1.0))
    is_upright = tilt_angle < np.pi / 6  # Within 30 degrees of vertical
    
    return vial_pos, vial_axis, tilt_angle, is_upright, vial_orient


def compute_optimal_grasp_point(vial, vial_pos, vial_axis, vial_height, tilt_angle):
    """
    Compute optimal grasp point along vial axis.
    
    For upright vials: grasp near top
    For tilted vials: grasp at center of mass for stability
    
    Returns:
        grasp_point: 3D position for gripper center
        grasp_height_ratio: 0=bottom, 0.5=middle, 1=top
    """
    if tilt_angle < np.pi / 6:  # Upright (< 30 degrees)
        # Grasp near top but not at extreme edge
        grasp_height_ratio = 0.75  # 75% up the vial
    elif tilt_angle < np.pi / 3:  # Moderately tilted (30-60 degrees)
        # Grasp at center for better stability
        grasp_height_ratio = 0.50
    else:  # Highly tilted or horizontal (> 60 degrees)
        # Grasp at center of mass
        grasp_height_ratio = 0.50
    
    # Calculate grasp point along vial axis
    # Start from vial center and move along axis
    offset_distance = (grasp_height_ratio - 0.5) * vial_height
    grasp_point = vial_pos + vial_axis * offset_distance
    
    return grasp_point, grasp_height_ratio


def compute_optimal_gripper_orientation(vial_axis, tilt_angle, approach_preference="axis_aligned"):
    """
    Compute optimal gripper orientation to grasp SHORT axis (diameter).
    
    Strategy: Gripper fingers should be PARALLEL to vial long axis,
    grasping the SHORT axis (diameter). This naturally maintains
    vertical orientation for placement.
    
    Args:
        vial_axis: Vial principal axis (long axis, cylinder axis)
        tilt_angle: Tilt from vertical
        approach_preference: "axis_aligned" (default), "top", or "adaptive"
    
    Returns:
        gripper_orient: Quaternion for gripper orientation
        approach_vector: Direction to approach from
        grasp_type: Description string
    """
    if approach_preference == "axis_aligned":
        # OPTIMAL: Gripper fingers PARALLEL to vial axis, grasp SHORT axis
        # This means gripper Y-axis (finger direction) aligns with vial axis
        
        # Normalize vial axis
        vial_axis_norm = vial_axis / np.linalg.norm(vial_axis)
        
        # Gripper coordinate frame:
        # - Y axis (fingers open/close direction) should align with vial axis
        # - Z axis (gripper approach) should be horizontal, perpendicular to vial
        # - X axis (gripper up) completes right-hand rule
        
        gripper_y = vial_axis_norm
        
        # Choose Z direction: horizontal and perpendicular to vial axis
        # Try [1, 0, 0] first, if parallel to vial try [0, 1, 0]
        candidate_z = np.array([1, 0, 0])
        if abs(np.dot(candidate_z, gripper_y)) > 0.9:  # Too parallel
            candidate_z = np.array([0, 1, 0])
        
        # Project to be perpendicular to gripper_y and normalize
        gripper_z = candidate_z - np.dot(candidate_z, gripper_y) * gripper_y
        gripper_z = gripper_z / np.linalg.norm(gripper_z)
        
        # Keep gripper_z horizontal (project to XY plane)
        gripper_z[2] = 0
        if np.linalg.norm(gripper_z) < 0.1:
            # Vial is horizontal, use perpendicular horizontal
            gripper_z = np.array([gripper_y[1], -gripper_y[0], 0])
        gripper_z = gripper_z / np.linalg.norm(gripper_z)
        
        # X axis: complete right-hand frame
        gripper_x = np.cross(gripper_y, gripper_z)
        gripper_x = gripper_x / np.linalg.norm(gripper_x)
        
        # Build rotation matrix [X, Y, Z] as columns
        rot_matrix = np.column_stack([gripper_x, gripper_y, gripper_z])
        gripper_orient = Rotation.from_matrix(rot_matrix).as_quat()
        
        # Approach from gripper Z direction (horizontal approach)
        approach_vector = gripper_z
        grasp_type = "Axis-aligned (fingers || vial axis)"
        
    elif approach_preference == "top":
        # Simple top-down grasp (old method)
        gripper_orient = Rotation.from_euler('xyz', [np.pi, 0, 0]).as_quat()
        approach_vector = np.array([0, 0, 1])
        grasp_type = "Top-down (simple)"
    
    else:  # adaptive
        # Choose based on tilt angle
        if tilt_angle < np.pi / 6:  # Upright
            # Use top-down
            gripper_orient = Rotation.from_euler('xyz', [np.pi, 0, 0]).as_quat()
            approach_vector = np.array([0, 0, 1])
            grasp_type = "Top-down (upright vial)"
        else:
            # Use axis-aligned for tilted
            return compute_optimal_gripper_orientation(vial_axis, tilt_angle, "axis_aligned")
    
    return gripper_orient, approach_vector, grasp_type


def compute_approach_clearance(grasp_point, approach_vector, all_vials, target_vial_idx, 
                               approach_distance=0.25):
    """
    Check if approach path is clear of obstacles.
    
    Returns:
        is_clear: True if path is clear
        clearance_score: 0-100, higher is better
        nearest_obstacle_dist: Distance to nearest obstacle along path
    """
    from robot_control import get_position
    
    # Sample points along approach path
    num_samples = 10
    min_clearance = float('inf')
    
    for t in np.linspace(0, 1, num_samples):
        sample_point = grasp_point + approach_vector * approach_distance * t
        
        # Check distance to all other vials
        for i, other_vial in enumerate(all_vials):
            if i == target_vial_idx:
                continue
            
            other_pos = get_position(other_vial)
            dist = np.linalg.norm(sample_point - other_pos)
            min_clearance = min(min_clearance, dist)
    
    # Score based on minimum clearance
    # Gripper width ~6cm, so need at least 4-5cm clearance
    if min_clearance > 0.08:
        clearance_score = 100.0
        is_clear = True
    elif min_clearance > 0.06:
        clearance_score = 80.0
        is_clear = True
    elif min_clearance > 0.04:
        clearance_score = 50.0
        is_clear = True
    else:
        clearance_score = 20.0
        is_clear = False
    
    return is_clear, clearance_score, min_clearance


def visualize_grasp_analysis(vial, grasp_point, approach_vector, gripper_orient, 
                             grasp_type, clearance_score, duration=10.0):
    """
    Visualize the grasp analysis with debug drawings.
    
    Shows:
    - Grasp point (sphere)
    - Approach vector (arrow)
    - Gripper orientation (coordinate frame)
    - Text labels
    """
    vial_pos = get_position(vial)
    
    # Draw grasp point as sphere
    if clearance_score > 70:
        color = [0, 1, 0]  # Green for good
    elif clearance_score > 40:
        color = [1, 1, 0]  # Yellow for okay
    else:
        color = [1, 0, 0]  # Red for poor
    
    # Grasp point marker
    p.addUserDebugText(
        text="GRASP",
        textPosition=grasp_point + [0, 0, 0.03],
        textColorRGB=color,
        textSize=1.0,
        lifeTime=duration
    )
    
    # Draw approach vector
    approach_end = grasp_point + approach_vector * 0.15
    p.addUserDebugLine(
        lineFromXYZ=grasp_point,
        lineToXYZ=approach_end,
        lineColorRGB=[0, 0, 1],
        lineWidth=3,
        lifeTime=duration
    )
    
    # Draw gripper coordinate frame at grasp point
    # X-axis (red), Y-axis (green), Z-axis (blue)
    R = Rotation.from_quat(gripper_orient)
    frame_length = 0.05
    
    for i, axis_color in enumerate([[1,0,0], [0,1,0], [0,0,1]]):
        axis_direction = R.apply(np.eye(3)[i])
        p.addUserDebugLine(
            lineFromXYZ=grasp_point,
            lineToXYZ=grasp_point + axis_direction * frame_length,
            lineColorRGB=axis_color,
            lineWidth=2,
            lifeTime=duration
        )
    
    # Draw vial axis for reference
    vial_axis = get_vial_axis(get_orientation(vial))
    p.addUserDebugLine(
        lineFromXYZ=vial_pos - vial_axis * 0.04,
        lineToXYZ=vial_pos + vial_axis * 0.04,
        lineColorRGB=[1, 0, 1],  # Magenta
        lineWidth=2,
        lifeTime=duration
    )
    
    # Text info
    info_text = f"{grasp_type}\nClearance: {clearance_score:.0f}"
    p.addUserDebugText(
        text=info_text,
        textPosition=grasp_point + [0.05, 0, 0.05],
        textColorRGB=[1, 1, 1],
        textSize=0.8,
        lifeTime=duration
    )


def analyze_and_visualize_grasp(vial, vial_idx, all_vials, vial_height, duration=10.0):
    """
    Complete grasp analysis and visualization for a vial.
    
    Returns:
        analysis_dict: Dictionary with all grasp information
    """
    # Analyze vial orientation
    vial_pos, vial_axis, tilt_angle, is_upright, vial_orient = analyze_vial_orientation(vial)
    
    # Compute optimal grasp point
    grasp_point, grasp_height_ratio = compute_optimal_grasp_point(
        vial, vial_pos, vial_axis, vial_height, tilt_angle
    )
    
    # Compute optimal gripper orientation
    gripper_orient, approach_vector, grasp_type = compute_optimal_gripper_orientation(
        vial_axis, tilt_angle, approach_preference="top"
    )
    
    # Check approach clearance
    is_clear, clearance_score, min_clearance = compute_approach_clearance(
        grasp_point, approach_vector, all_vials, vial_idx
    )
    
    # For visualization, use axis_aligned to show optimal theoretical grasp
    # But actual execution uses top-down for reliability
    gripper_orient_vis, approach_vector_vis, grasp_type_vis = compute_optimal_gripper_orientation(
        vial_axis, tilt_angle, approach_preference="axis_aligned"
    )
    
    # Visualize the theoretical optimal grasp
    visualize_grasp_analysis(
        vial, grasp_point, approach_vector_vis, gripper_orient_vis,
        grasp_type_vis, clearance_score, duration
    )
    
    # Return analysis
    return {
        'vial_pos': vial_pos,
        'vial_axis': vial_axis,
        'tilt_angle': np.degrees(tilt_angle),
        'is_upright': is_upright,
        'grasp_point': grasp_point,
        'grasp_height_ratio': grasp_height_ratio,
        'gripper_orient': gripper_orient,
        'approach_vector': approach_vector,
        'grasp_type': grasp_type,
        'approach_clear': is_clear,
        'clearance_score': clearance_score,
        'min_clearance': min_clearance
    }

