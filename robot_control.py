"""
Robot control functions (Simple version)
"""
import mengine as m
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from config import *

# Collision checking state
_collision_check_enabled = True
_last_collision_check = None

def get_position(obj):
    """Get object position - handles both mengine Shape and PyBullet body ID"""
    if isinstance(obj, int):
        # PyBullet body ID (for real tube URDF)
        pos, _ = p.getBasePositionAndOrientation(obj)
    else:
        # mengine Shape object
        pos, _ = obj.get_base_pos_orient()
    return np.array(pos)

def get_orientation(obj):
    """Get object orientation - handles both mengine Shape and PyBullet body ID"""
    if isinstance(obj, int):
        # PyBullet body ID (for real tube URDF)
        _, orient = p.getBasePositionAndOrientation(obj)
    else:
        # mengine Shape object
        _, orient = obj.get_base_pos_orient()
    return orient

def get_vial_axis(orientation):
    """Get vial principal axis from orientation"""
    rot = Rotation.from_quat(orientation)
    rot_matrix = rot.as_matrix()
    return rot_matrix[:, 2]

def check_robot_collisions(robot, box_pieces=None, table=None, held_vial=None, verbose=False):
    """
    Check for collisions between robot and environment (box, table).
    
    IMPORTANT: Vial collisions are NORMAL and EXPECTED (we're grasping them).
    We only check for dangerous collisions:
    - Robot arm/gripper with box walls
    - Robot arm/gripper with table
    - Self-collision (robot links)
    
    Args:
        robot: Robot object
        box_pieces: List of box wall pieces (to check collision with)
        table: Table object (to check collision with)
        held_vial: Currently held vial (not used, kept for API compatibility)
        verbose: Print collision details
    
    Returns:
        dict with collision info: {
            'has_collision': bool,
            'self_collision': bool,
            'box_collision': bool,
            'table_collision': bool,
            'collision_count': int
        }
    """
    global _last_collision_check
    
    if not _collision_check_enabled:
        return {'has_collision': False, 'self_collision': False, 
                'box_collision': False, 'table_collision': False, 'collision_count': 0}
    
    collision_info = {
        'has_collision': False,
        'self_collision': False,
        'box_collision': False,
        'table_collision': False,
        'collision_count': 0
    }
    
    robot_body = robot.robot
    
    # Check self-collision (robot links colliding with each other)
    # This is ALWAYS dangerous
    # IMPORTANT: Only check robot-to-robot collisions, not robot-to-vials!
    for i in range(p.getNumJoints(robot_body)):
        # Specify bodyB=robot_body to ONLY get robot-to-robot contacts
        contacts = p.getContactPoints(bodyA=robot_body, linkIndexA=i, bodyB=robot_body)
        for contact in contacts:
            linkA = contact[3]
            linkB = contact[4]
            
            # Filter adjacent links (normal articulation)
            if abs(linkB - linkA) <= 1:
                continue
            
            collision_info['self_collision'] = True
            collision_info['has_collision'] = True
            collision_info['collision_count'] += 1
            if verbose:
                print(f"    Self-collision: link {linkA} <-> link {linkB}")
    
    # Check collision with BOX WALLS (dangerous!)
    # IMPORTANT: Skip box_pieces[0] which is the BOTTOM (gripper needs to enter box)
    # Only check walls (pieces[1:4])
    if box_pieces is not None and len(box_pieces) > 1:
        for idx, box_piece in enumerate(box_pieces[1:], start=1):  # Skip bottom, only check walls
            box_id = box_piece if isinstance(box_piece, int) else box_piece.body
            contacts = p.getContactPoints(bodyA=robot_body, bodyB=box_id)
            
            if len(contacts) > 0:
                collision_info['box_collision'] = True
                collision_info['has_collision'] = True
                collision_info['collision_count'] += len(contacts)
                if verbose:
                    print(f"    Box wall {idx} collision: {len(contacts)} contact points")
    
    # Check collision with TABLE (dangerous!)
    if table is not None:
        table_id = table if isinstance(table, int) else table.body
        contacts = p.getContactPoints(bodyA=robot_body, bodyB=table_id)
        
        if len(contacts) > 0:
            collision_info['table_collision'] = True
            collision_info['has_collision'] = True
            collision_info['collision_count'] += len(contacts)
            if verbose:
                print(f"    Table collision: {len(contacts)} contact points")
    
    _last_collision_check = collision_info
    return collision_info

def compute_grasp_orientation(vial_pos, vial_orient):
    """
    Compute grasp orientation for a vial.
    
    STRATEGY: 
    - Vertical vials: top-down grasp
    - Horizontal vials: grasp from ABOVE at the highest point of the cylinder
    
    Key: Always approach from above (Z+) for collision avoidance
    """
    vial_axis = get_vial_axis(vial_orient)  # Vial LONG axis
    
    # Calculate tilt
    vertical = np.array([0, 0, 1])
    tilt_angle = np.arccos(np.clip(abs(np.dot(vial_axis, vertical)), 0, 1))
    tilt_degrees = np.degrees(tilt_angle)
    
    # Project vial axis to XY plane to find rotation
    vial_xy = np.array([vial_axis[0], vial_axis[1], 0])
    vial_xy_norm = np.linalg.norm(vial_xy)
    
    if vial_xy_norm < 0.01 or tilt_degrees < 30:
        # Vial is nearly vertical, simple top-down grasp
        approach_dir = np.array([0, 0, 1])
        grasp_pos = np.array(vial_pos) + np.array([0, 0, 0.005])
        gripper_orient = m.get_quaternion([np.pi, 0, 0])
        grasp_type = "Top-down (upright)"
    else:
        # Vial is tilted/horizontal
        # SIMPLIFIED: Just grasp at vial center height
        # The key is correct gripper orientation, not position offset
        
        vial_xy_unit = vial_xy / vial_xy_norm
        
        # Grasp at vial center (no vertical offset for horizontal vials)
        # The gripper will approach from above and close around the cylinder
        grasp_pos = np.array(vial_pos).copy()
        
        # Approach from directly above
        approach_dir = np.array([0, 0, 1])
        
        # Gripper orientation: fingers perpendicular to vial long axis
        # Find SHORT axis direction (perpendicular to vial axis in XY plane)
        short_axis_xy = np.array([-vial_xy_unit[1], vial_xy_unit[0], 0])  # 90° rotation in XY
        
        # Gripper configuration for top-down approach:
        # - Z: points down [0, 0, -1]
        # - X: fingers close along SHORT axis (perpendicular to vial long axis)  
        # - Y: completes frame
        
        gripper_z = np.array([0, 0, -1])
        gripper_x = short_axis_xy  # Fingers close along diameter
        gripper_y = np.cross(gripper_z, gripper_x)
        gripper_y = gripper_y / np.linalg.norm(gripper_y)
        
        # Build rotation matrix
        rot_matrix = np.column_stack([gripper_x, gripper_y, gripper_z])
        gripper_orient = Rotation.from_matrix(rot_matrix).as_quat()
        
        rotation_z = np.arctan2(short_axis_xy[1], short_axis_xy[0])
        grasp_type = f"Top grasp horizontal (rot {np.degrees(rotation_z):.0f}°)"
    
    return grasp_pos, gripper_orient, approach_dir, grasp_type

def open_gripper(robot, vial_radius, is_horizontal=False):
    """Open gripper based on vial size and orientation
    
    Args:
        vial_radius: Vial radius in meters
        is_horizontal: If True, open wider for horizontal vial grasp
    """
    if is_horizontal:
        # FIXED: Horizontal vials need wider opening, but not too wide
        # 3.0x radius ensures fingers make contact (was 4.0x, too wide)
        open_amount = min(vial_radius * 3.0, 0.04)  # 3x radius, max 40mm
    else:
        # Vertical vials: normal opening
        open_amount = min(vial_radius * GRIPPER_OPEN_MULTIPLIER, 0.035)  # Max 35mm
    
    robot.set_gripper_position([open_amount, open_amount], 
                               set_instantly=False, 
                               force=300)

def close_gripper_gently(robot, vial_radius):
    """Close gripper gently around vial"""
    # Close to slightly smaller than vial radius (to grip the sides)
    close_amount = vial_radius * GRIPPER_CLOSE_MULTIPLIER
    robot.set_gripper_position([close_amount, close_amount], 
                               set_instantly=False, 
                               force=GRIPPER_FORCE)

def move_to_pose(robot, target_pos, target_orient, steps=700, tolerance=0.05, 
                 check_collisions=False, box_pieces=None, table=None, held_vial=None):
    """
    Move robot to target pose using IK.
    
    Args:
        robot: Robot object
        target_pos: Target position [x, y, z]
        target_orient: Target orientation quaternion
        steps: Number of simulation steps
        tolerance: Joint angle tolerance for convergence
        check_collisions: If True, check for collisions during movement
        box_pieces: List of box pieces (for collision checking)
        table: Table object (for collision checking)
        held_vial: Currently held vial (not used, kept for API compatibility)
    
    Returns:
        bool: True if successful, False if failed or collision detected
    """
    try:
        q_target = robot.ik(robot.end_effector, target_pos, target_orient, 
                           max_iterations=500)
        if q_target is None:
            print(f"    [IK FAILED] Cannot reach target pose")
            print(f"      Target pos: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            from scipy.spatial.transform import Rotation as R
            euler = R.from_quat(target_orient).as_euler('xyz', degrees=True)
            print(f"      Target orient (euler): [{euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°]")
            return False
        
        for i in range(steps):
            robot.control(q_target[:7])
            m.step_simulation()
            
            # Periodic collision check
            if check_collisions and i % 30 == 0:
                from config import COLLISION_THRESHOLD
                collision_info = check_robot_collisions(robot, box_pieces, table, held_vial, verbose=True)
                if collision_info['has_collision']:
                    # Filter out minor/transient collisions using threshold
                    if collision_info['collision_count'] > COLLISION_THRESHOLD:
                        print(f"    Collision detected: {collision_info['collision_count']} contacts (threshold={COLLISION_THRESHOLD})")
                        if collision_info['box_collision']:
                            print(f"      - Box wall collision")
                        if collision_info['table_collision']:
                            print(f"      - Table collision")
                        if collision_info['self_collision']:
                            print(f"      - Self collision")
                        return False
                    else:
                        # Debug: show why we're NOT aborting
                        if i == 30:  # Only print once
                            print(f"    [DEBUG] Collision below threshold: {collision_info['collision_count']} <= {COLLISION_THRESHOLD}")
            
            if i % 120 == 0:
                current_q = robot.get_joint_angles()[:7]
                error = np.linalg.norm(current_q - q_target[:7])
                if error < tolerance:
                    return True
        return True
    except:
        return False

def create_vial_constraint(robot, vial):
    """Create constraint to attach vial to gripper with snap-to-center stabilization"""
    try:
        from config import GRIPPER_CENTER_Y_OFFSET
        
        # Get vial body ID
        if isinstance(vial, int):
            vial_body_id = vial
        else:
            vial_body_id = vial.body
        
        # Get end effector position
        ee_link_id = robot.end_effector
        ee_pos, ee_orient = robot.get_link_pos_orient(ee_link_id)
        vial_pos, vial_orient = p.getBasePositionAndOrientation(vial_body_id)
        
        # Check if vial is reasonably close to gripper
        dist_to_ee = np.linalg.norm(np.array(vial_pos) - np.array(ee_pos))
        
        print(f"     Grasp check: dist to end effector={dist_to_ee*1000:.1f}mm")
        print(f"     End effector: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
        print(f"     Vial pos: [{vial_pos[0]:.3f}, {vial_pos[1]:.3f}, {vial_pos[2]:.3f}]")
        
        # Validation: vial should be within ~10cm of end effector
        # INCREASED from 8cm to 10cm to allow for larger vials and slight position errors
        MAX_DIST = 0.10
        if dist_to_ee > MAX_DIST:
            print(f"     Vial too far ({dist_to_ee*1000:.0f}mm > {MAX_DIST*1000:.0f}mm) - NO constraint")
            return None
        
        # NO SNAP - Keep vial at current position
        # Create constraint based on current relative position
        print(f"     Creating constraint (position locked, orientation FREE for re-orienting)...")
        
        # Calculate relative position from end effector to vial
        rel_pos = np.array(vial_pos) - np.array(ee_pos)
        
        # Create constraint that locks ONLY position, NOT orientation
        # This allows vial to be re-oriented during lift via gravity + gripper fingers
        # The gripper fingers will guide the vial to vertical as we lift
        constraint_id = p.createConstraint(
            parentBodyUniqueId=robot.body,
            parentLinkIndex=ee_link_id,
            childBodyUniqueId=vial_body_id,
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,  # Point-to-point = position only
            jointAxis=[0, 0, 0],
            parentFramePosition=rel_pos.tolist(),
            childFramePosition=[0, 0, 0]
            # NO parentFrameOrientation/childFrameOrientation = orientation is FREE
        )
        
        return constraint_id
        
    except Exception as e:
        print(f"     Constraint failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def remove_constraint(constraint_id):
    """Remove constraint"""
    if constraint_id is not None:
        try:
            p.removeConstraint(constraint_id)
            return True
        except:
            pass
    return False

def singulation_push(robot, push_start, push_end, push_orient, steps=400):
    """
    Perform a pushing motion to separate stacked/piled vials.
    
    Args:
        robot: Robot object
        push_start: Starting position for push [x, y, z]
        push_end: Ending position for push [x, y, z]
        push_orient: Gripper orientation during push (quaternion)
        steps: Number of simulation steps for the push
    
    Returns:
        success: True if push completed
    """
    # Close gripper FULLY to form a solid "blade" for pushing
    robot.set_gripper_position([0.0, 0.0], set_instantly=False, force=1000)
    for _ in range(100):  # More time to close fully
        m.step_simulation()
    
    # Move to push start
    if not move_to_pose(robot, push_start, push_orient, steps=300):
        return False
    
    # VERIFY: Check if gripper reached push_start inside box
    import pybullet as p
    from config import BOX_POS, BOX_SIZE
    
    ee_state = p.getLinkState(robot.body, robot.end_effector)
    ee_pos = ee_state[0]
    box_center = np.array(BOX_POS)
    box_half = np.array(BOX_SIZE) / 2
    ARM_CLEARANCE = 0.08  # 8cm clearance (reduced for better edge access)
    
    outside_x = abs(ee_pos[0] - box_center[0]) > box_half[0] - ARM_CLEARANCE
    outside_y = abs(ee_pos[1] - box_center[1]) > box_half[1] - ARM_CLEARANCE
    
    if outside_x or outside_y:
        # FIXED: Print both planned and actual positions
        print(f"     ⚠️  Push start failed - gripper outside box")
        print(f"        Planned: [{push_start[0]:.3f}, {push_start[1]:.3f}]")
        print(f"        Actual:  [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}]")
        print(f"        Box bounds: X=[{box_center[0]-box_half[0]+ARM_CLEARANCE:.3f}, "
              f"{box_center[0]+box_half[0]-ARM_CLEARANCE:.3f}], "
              f"Y=[{box_center[1]-box_half[1]+ARM_CLEARANCE:.3f}, "
              f"{box_center[1]+box_half[1]-ARM_CLEARANCE:.3f}]")
        return False
    
    # Execute push motion with moderate force
    # Key: maintain contact without getting stuck on floor
    import pybullet as p
    from config import BOX_POS, BOX_SIZE
    
    # Box boundaries for validation
    box_center = np.array(BOX_POS)
    box_half = np.array(BOX_SIZE) / 2
    ARM_CLEARANCE = 0.08  # 8cm clearance (reduced for better edge access)
    
    for i in range(steps):
        t = i / steps
        current_pos = np.array(push_start) * (1 - t) + np.array(push_end) * t
        
        # Keep at target height (don't lower - we already set it correctly)
        # No adjustment needed
        
        q_target = robot.ik(robot.end_effector, current_pos, push_orient, max_iterations=100)
        if q_target is not None:
            robot.control(q_target[:7])
        
        # VERIFY: Check if gripper is inside box every 50 steps
        if i % 50 == 0:
            ee_state = p.getLinkState(robot.body, robot.end_effector)
            ee_pos = ee_state[0]
            
            # Check if gripper is outside box
            outside_x = abs(ee_pos[0] - box_center[0]) > box_half[0] - ARM_CLEARANCE
            outside_y = abs(ee_pos[1] - box_center[1]) > box_half[1] - ARM_CLEARANCE
            
            if outside_x or outside_y:
                print(f"     ⚠️  Push aborted: Gripper outside box at [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}]")
                return False
        
        # Apply MODERATE downward force to maintain contact (not too much!)
        if i % 10 == 0:  # Less frequent
            # Get end effector link
            ee_state = p.getLinkState(robot.body, robot.end_effector)
            ee_pos = ee_state[0]
            
            # Apply gentle downward force (reduced from 50N to 20N)
            p.applyExternalForce(
                robot.body,
                robot.end_effector,
                forceObj=[0, 0, -20],  # 20N downward (gentle)
                posObj=ee_pos,
                flags=p.WORLD_FRAME
            )
        
        m.step_simulation()
    
    # Hold at end position briefly to ensure push completes
    for _ in range(100):
        m.step_simulation()
    
    return True

def calculate_push_trajectory(vial_pos, nearest_vial_pos, push_distance=0.08):
    """
    Calculate push trajectory to separate target vial from nearest neighbor.
    
    Strategy: Push the nearest vial AWAY from the target at low height.
    Ensures push positions stay inside the box.
    
    Args:
        vial_pos: Target vial position
        nearest_vial_pos: Nearest neighbor position
        push_distance: How far to push (default 8cm)
    
    Returns:
        (push_start, push_end, push_orient) or None if can't calculate
    """
    from config import TABLE_HEIGHT, BOX_BOTTOM_THICKNESS, BOX_POS, BOX_SIZE
    
    # Direction from target to nearest (we'll push nearest away)
    direction = np.array(nearest_vial_pos) - np.array(vial_pos)
    direction[2] = 0  # Keep horizontal
    dist = np.linalg.norm(direction)
    
    if dist < 0.01:
        return None
    
    direction = direction / dist
    
    # Push height: Balance between contact and clearance
    # - Too low (1cm): gripper hits floor, can't move
    # - Too high (5cm): misses vials
    # - Sweet spot: 2.5-3cm (middle of small vial radius ~8mm, height ~35mm)
    # For vials lying down, center is ~1-2cm above floor
    push_height = TABLE_HEIGHT + BOX_BOTTOM_THICKNESS + 0.025  # 2.5cm above floor
    
    # Box boundaries - CRITICAL: account for ENTIRE robot arm geometry
    # Not just gripper! Need to consider:
    # - Gripper fingers: ~6cm wide (closed)
    # - End effector body: ~8cm wide rectangular block
    # - Wrist joint: ~6cm diameter
    # Total clearance needed: ~8cm for safe operation
    
    ARM_CLEARANCE = 0.08  # 8cm clearance from walls for arm/end effector (reduced for better edge access)
    
    box_x_min = BOX_POS[0] - BOX_SIZE[0]/2 + ARM_CLEARANCE
    box_x_max = BOX_POS[0] + BOX_SIZE[0]/2 - ARM_CLEARANCE
    box_y_min = BOX_POS[1] - BOX_SIZE[1]/2 + ARM_CLEARANCE
    box_y_max = BOX_POS[1] + BOX_SIZE[1]/2 - ARM_CLEARANCE
    
    # Check if push is even possible with these constraints
    available_space_x = box_x_max - box_x_min
    available_space_y = box_y_max - box_y_min
    
    if available_space_x < 0.15 or available_space_y < 0.15:
        print(f"     ⚠️  Box too small for safe pushing (available: {available_space_x*100:.0f}x{available_space_y*100:.0f}cm)")
        return None
    
    # Push start: slightly before the nearest vial (5cm back for approach)
    push_start = np.array(nearest_vial_pos) - direction * 0.05
    push_start[2] = push_height
    
    # Push end: push the nearest vial away
    push_end = np.array(nearest_vial_pos) + direction * push_distance
    push_end[2] = push_height
    
    # EARLY CHECK: If nearest vial is too close to edge, push is not feasible
    # Check if nearest vial has enough space for push trajectory
    nearest_margin_x = abs(nearest_vial_pos[0] - BOX_POS[0])
    nearest_margin_y = abs(nearest_vial_pos[1] - BOX_POS[1])
    box_half_x = BOX_SIZE[0] / 2
    box_half_y = BOX_SIZE[1] / 2
    
    # Need at least 15cm space (5cm approach + 8cm push + 2cm margin)
    MIN_PUSH_SPACE = 0.15
    space_x = box_half_x - nearest_margin_x
    space_y = box_half_y - nearest_margin_y
    
    if space_x < MIN_PUSH_SPACE or space_y < MIN_PUSH_SPACE:
        print(f"     ⚠️  Vial too close to edge for push (space: X={space_x*100:.0f}cm, Y={space_y*100:.0f}cm)")
        return None
    
    # Check if original trajectory would go out of bounds
    start_clamped = (push_start[0] < box_x_min or push_start[0] > box_x_max or 
                     push_start[1] < box_y_min or push_start[1] > box_y_max)
    end_clamped = (push_end[0] < box_x_min or push_end[0] > box_x_max or 
                   push_end[1] < box_y_min or push_end[1] > box_y_max)
    
    # If push would be heavily clamped, reject early
    if start_clamped and end_clamped:
        print(f"     ⚠️  Push trajectory entirely outside box bounds")
        return None
    
    # Clamp positions to stay inside box
    push_start[0] = np.clip(push_start[0], box_x_min, box_x_max)
    push_start[1] = np.clip(push_start[1], box_y_min, box_y_max)
    push_end[0] = np.clip(push_end[0], box_x_min, box_x_max)
    push_end[1] = np.clip(push_end[1], box_y_min, box_y_max)
    
    # Check if push trajectory is valid (not too short after clamping)
    push_dist = np.linalg.norm(push_end[:2] - push_start[:2])
    if push_dist < 0.03:  # Less than 3cm after clamping
        print(f"     Push trajectory too short after clamping ({push_dist*100:.1f}cm)")
        return None
    
    # Warn if clamping significantly reduced push effectiveness
    if end_clamped and push_dist < 0.05:
        print(f"     WARNING: Push limited by box boundary (only {push_dist*100:.1f}cm)")
    
    # Gripper orientation: pointing down, aligned with push direction
    push_orient = m.get_quaternion([np.pi, 0, 0])
    
    # VALIDATION: Check if approach path (from above) would collide with box walls
    # Approach from 20cm above push_start
    approach_start = push_start.copy()
    approach_start[2] += 0.20  # 20cm above
    
    # Check if approach position is inside box XY bounds (with clearance)
    if not (box_x_min <= approach_start[0] <= box_x_max and
            box_y_min <= approach_start[1] <= box_y_max):
        print(f"     ⚠️  Approach path would collide with box walls")
        print(f"        Approach XY: ({approach_start[0]:.3f}, {approach_start[1]:.3f})")
        print(f"        Box bounds: X[{box_x_min:.3f}, {box_x_max:.3f}], Y[{box_y_min:.3f}, {box_y_max:.3f}]")
        return None
    
    return push_start, push_end, push_orient

def calculate_vertical_placement_pose(robot, vial, target_vial_center):
    """
    Calculate gripper pose to place vial vertically at target position.
    
    This accounts for the offset between gripper center and vial center,
    ensuring the vial ends up vertical at the target location.
    
    Args:
        robot: Robot object
        vial: Vial object (to get current relative position)
        target_vial_center: Desired vial center position [x, y, z]
    
    Returns:
        (gripper_pos, gripper_orient): Pose for gripper to achieve target
    """
    from scipy.spatial.transform import Rotation as R
    
    # Get current vial position
    vial_pos = get_position(vial)
    
    # Get current end effector position
    ee_state = p.getLinkState(robot.body, robot.end_effector)
    ee_pos = np.array(ee_state[0])
    
    # Calculate current offset from gripper to vial
    offset = vial_pos - ee_pos
    
    # Target gripper position = target vial position - offset
    # But we want the vial VERTICAL, so we only keep XY offset
    # and set Z offset to maintain grip height
    target_gripper_pos = target_vial_center - np.array([offset[0], offset[1], 0.05])
    
    # Target orientation: vertical down (for vertical vial placement)
    target_orient = m.get_quaternion([np.pi, 0, 0])
    
    return target_gripper_pos, target_orient

def shake_box_with_gripper(robot, box_pieces, num_shakes=3, shake_distance=0.04):
    """
    Shake vials by sweeping gripper through the box to redistribute them.
    
    This helps redistribute vials from borders to center without spilling.
    
    Strategy: Since box is static, we sweep the gripper through the box at vial
    height to push vials toward the center, creating better grasping opportunities.
    
    Args:
        robot: Robot object
        box_pieces: List of box wall body IDs (not used, kept for compatibility)
        num_shakes: Number of sweep passes
        shake_distance: Sweep distance (meters)
    
    Returns:
        success: True if shake completed
    """
    import mengine as m
    import pybullet as p
    
    print("  🔄 Sweeping gripper to redistribute vials...")
    
    # Get box dimensions from config
    from config import BOX_POS, BOX_SIZE, TABLE_HEIGHT, BOX_BOTTOM_THICKNESS
    box_center = np.array(BOX_POS)
    box_width = BOX_SIZE[0]
    box_depth = BOX_SIZE[1]
    
    # Vial height: slightly above box floor
    vial_height = TABLE_HEIGHT + BOX_BOTTOM_THICKNESS + 0.03  # 3cm above floor
    
    # Gripper orientation: vertical down with fingers closed
    sweep_orient = m.get_quaternion([np.pi, 0, 0])
    
    print(f"     Performing {num_shakes} sweep passes to redistribute vials...")
    
    # Close gripper to make it a solid "paddle" for sweeping
    robot.set_gripper_position([0.0, 0.0], set_instantly=False, force=100)
    for _ in range(50):
        m.step_simulation()
    
    # Perform multiple sweep passes in different directions
    sweep_patterns = [
        # (start_offset, end_offset, description)
        ([0.15, 0.15], [-0.10, -0.10], "corner-to-center diagonal"),
        ([-0.15, 0.15], [0.10, -0.10], "opposite diagonal"),
        ([0.15, 0.0], [-0.10, 0.0], "right-to-center horizontal"),
        ([0.0, 0.15], [0.0, -0.10], "top-to-center vertical"),
    ]
    
    for sweep_idx in range(min(num_shakes, len(sweep_patterns))):
        start_offset, end_offset, description = sweep_patterns[sweep_idx]
        
        # Start position: near box edge
        start_pos = np.array([
            box_center[0] + start_offset[0],
            box_center[1] + start_offset[1],
            vial_height
        ])
        
        # End position: toward box center
        end_pos = np.array([
            box_center[0] + end_offset[0],
            box_center[1] + end_offset[1],
            vial_height
        ])
        
        print(f"     Sweep {sweep_idx+1}: {description}")
        
        # Move to start position (approach from above)
        approach_start = start_pos + np.array([0, 0, 0.15])
        q_approach = robot.ik(robot.end_effector, approach_start, sweep_orient, max_iterations=150)
        if q_approach is None:
            print(f"       Cannot reach start position, skipping this sweep")
            continue
        
        robot.control(q_approach)
        for _ in range(150):
            m.step_simulation()
        
        # Descend to sweep height
        q_start = robot.ik(robot.end_effector, start_pos, sweep_orient, max_iterations=150)
        if q_start is None:
            print(f"       Cannot reach sweep height, skipping this sweep")
            continue
        
        robot.control(q_start)
        for _ in range(150):
            m.step_simulation()
        
        # Sweep to end position (push vials)
        q_end = robot.ik(robot.end_effector, end_pos, sweep_orient, max_iterations=150)
        if q_end is not None:
            robot.control(q_end)
            for _ in range(300):  # Slow sweep to push vials effectively
                m.step_simulation()
        
        # Lift up
        approach_end = end_pos + np.array([0, 0, 0.15])
        q_lift = robot.ik(robot.end_effector, approach_end, sweep_orient, max_iterations=150)
        if q_lift is not None:
            robot.control(q_lift)
            for _ in range(150):
                m.step_simulation()
    
    # 8. Return to home position
    home_joints = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    robot.control(home_joints)
    for _ in range(200):
        m.step_simulation()
    
    # 9. Let vials settle after shaking
    print("     Settling vials after shake...")
    for _ in range(800):  # Longer settling time
        m.step_simulation()
    
    print("     ✓ Box shaken successfully, vials redistributed")
    return True

