#!/usr/bin/env python3
"""
Complete vial singulation system (Simple version with modular structure)

Features:
- 3 different vial sizes (small, medium, large)
- 3 matching trays with appropriate hole sizes
- Real-time CV tracking (position updates)
- Intelligent grasping adapted to vial size
- Constraint-based attachment for secure transport
"""
import mengine as m
import numpy as np
import time
import random
import pybullet as p

# Set random seed for reproducibility (different seed each run)
random_seed = int(time.time() * 1000) % (2**31)  # Use timestamp as seed
random.seed(random_seed)
np.random.seed(random_seed)
print(f"Random seed: {random_seed}")

# Note: PyBullet's physics engine also has some randomness in contact resolution
# and friction, but it doesn't have a direct random seed API. The randomness
# comes from the initial drop positions and orientations set above.

# Import modules
from config import *
from scene_builder import create_scene
from robot_control import (
    get_position, get_orientation, get_vial_axis,
    compute_grasp_orientation,
    open_gripper, close_gripper_gently,
    move_to_pose, check_robot_collisions,
    create_vial_constraint, remove_constraint,
    singulation_push, calculate_push_trajectory,
    calculate_vertical_placement_pose,
    shake_box_with_gripper
)
from task_logic import (
    compute_target_slot_position,
    check_placement_success,
    is_tray_full,
    check_grasp_isolation,
    compute_isolation_score,
    visualize_isolation_scores,
    find_most_isolated_vial
)
from grasp_analysis import analyze_and_visualize_grasp
from experiment_logger import ExperimentLogger, get_logger, reset_logger

# Import multi-candidate system if enabled
if USE_MULTI_CANDIDATE_GRASPS:
    print("Using MULTI-CANDIDATE grasp system")
    from grasp_candidates import (
        generate_grasp_candidates,
        select_best_grasp
    )
else:
    print("Using SIMPLE grasp system")

# Helper function for simulation
def sim_wait(steps=100):
    """Wait for simulation - real-time or step-by-step"""
    if REAL_TIME_MODE:
        # Real-time: just wait
        time.sleep(steps / 100.0)  # Convert steps to seconds (100 steps = 1s)
    else:
        # Step-by-step: run steps
        for _ in range(steps):
            m.step_simulation()

# ========== SCENE CONFIGURATION ==========
# Set to True to use predefined test scene for border/angle testing
# Set to False to use random vial placement
USE_TEST_SCENE = False  # Change to False for random scene

print("="*70)
print("COMPLETE VIAL SINGULATION SYSTEM - Simple Algorithm")
print("="*70)
if USE_TEST_SCENE:
    print("MODE: TEST SCENE (predefined positions for border/angle testing)")
else:
    print("MODE: RANDOM SCENE (30 random vials)")
print(f"\n BOX: {BOX_SIZE[0]*100:.0f}cm x {BOX_SIZE[1]*100:.0f}cm x {BOX_SIZE[2]*100:.0f}cm")
print(f"   Bottom: {BOX_BOTTOM_THICKNESS*100:.1f}cm thick")
print(f"\nVIALS:")
for vtype, spec in VIAL_SPECS.items():
    print(f"   {vtype.upper()}: {spec['count']}x, r={spec['radius']*1000:.0f}mm, "
          f"h={spec['height']*1000:.0f}mm, m={spec['mass']*1000:.0f}g")
print(f"\nTRAYS: 3 trays with matching hole sizes")
for vtype, config in TRAY_CONFIGS.items():
    print(f"   {vtype.upper()}: {config['slot_size']*100:.1f}cm slots, "
          f"wall height={config.get('wall_height', TRAY_WALL_HEIGHT)*100:.1f}cm")

# Create scene (use real tube URDF for large vials, optionally use test scene)
env, vials, vial_types, box_pieces, table = create_scene(use_real_tube_model=True, 
                                       use_test_scene=USE_TEST_SCENE,
                                       render=True)  # Enable GUI

print("[6/7] Creating robot...")
robot = m.Robot.Panda(position=[0, 0, 0.76])
robot.motor_gains = 0.03  # Moderate speed

# Enable real-time simulation if configured
if REAL_TIME_MODE:
    print("⚡ Enabling REAL-TIME simulation mode...")
    p.setRealTimeSimulation(1)
    p.setTimeStep(TIME_STEP)
    print(f"   Time step: {TIME_STEP*1000:.2f}ms ({1/TIME_STEP:.0f} Hz)")
else:
    print(" Using step-by-step simulation mode (accurate but slow)")

print("[7/7] Settling physics...")
if REAL_TIME_MODE:
    # Real-time: just wait
    print("   Waiting 3 seconds for physics to settle...")
    time.sleep(3.0)
else:
    # Step-by-step: run steps
    print("   Waiting 3 seconds for physics to settle...")
    for i in range(2000):  # 20 seconds
        m.step_simulation()
        if i % 400 == 0:
            print(f"   {i/100:.1f}s...")

print("\n Scene ready!")

# ========== WORKSPACE VERIFICATION ==========
print("\n[Verifying tray reachability...]")
test_orient = m.get_quaternion([np.pi, 0, 0])
all_reachable = True
min_distance = float('inf')
max_distance = 0.0
robot_base_pos = np.array([0, 0, 0.76])

for vtype in ['small', 'medium', 'large']:
    tray_config = TRAY_CONFIGS[vtype]
    rows = tray_config['rows']
    cols = tray_config['cols']
    
    print(f"   {vtype.upper()} tray:")
    
    for row in range(rows):
        for col in range(cols):
            slot_pos = compute_target_slot_position(vtype, row, col)
            test_pos = [slot_pos[0], slot_pos[1], slot_pos[2] + 0.30]
            
            # Calculate distance from robot base
            dist_xy = np.linalg.norm(np.array(slot_pos[:2]))
            min_distance = min(min_distance, dist_xy)
            max_distance = max(max_distance, dist_xy)
            
            # Test IK for above position
            q_test = robot.ik(robot.end_effector, test_pos, test_orient, max_iterations=100)
            
            # Also test actual descend position
            q_descend = robot.ik(robot.end_effector, slot_pos, test_orient, max_iterations=100)
            
            if q_test is None or q_descend is None:
                status = "FAIL"
                all_reachable = False
                print(f"     Slot ({row},{col}): {dist_xy*100:.1f}cm from robot - UNREACHABLE (IK fail)")
                print(f"       Position: [{slot_pos[0]:.3f}, {slot_pos[1]:.3f}, {slot_pos[2]:.3f}]")
            else:
                # Move robot to descend position and check for collisions
                robot.control(q_descend[:7])
                sim_wait(100)
                
                # Check for collisions with robot arm
                collision_detected = False
                contact_points = p.getContactPoints(bodyA=robot.body)
                
                for contact in contact_points:
                    # Check if robot arm collides with anything (table, tray, etc)
                    bodyB = contact[2]
                    linkA = contact[3]
                    
                    # Ignore gripper finger contacts (links 9, 10, 11) - they're expected
                    if linkA in [9, 10, 11]:
                        continue
                    
                    # Collision detected with arm/wrist (links 0-8)
                    if linkA >= 0 and linkA <= 8:
                        collision_detected = True
                        break
                
                # Check joint limits
                current_joints = robot.get_joint_angles()[:7]
                joint_limits = [
                    (-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973),
                    (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973)
                ]
                
                margins = []
                for angle, (min_lim, max_lim) in zip(current_joints, joint_limits):
                    margin = min(angle - min_lim, max_lim - angle)
                    margins.append(margin)
                
                min_margin = min(margins)
                
                if collision_detected:
                    status = "FAIL"
                    all_reachable = False
                    print(f"     Slot ({row},{col}): {dist_xy*100:.1f}cm - UNREACHABLE (ARM COLLISION)")
                elif min_margin < 0.1:  # Within 6° of limit
                    status = "MARGINAL"
                    print(f"     Slot ({row},{col}): {dist_xy*100:.1f}cm - REACHABLE but tight (margin: {min_margin:.3f} rad)")
                else:
                    status = "OK"
                    print(f"     Slot ({row},{col}): {dist_xy*100:.1f}cm - OK")

print(f"\n   Distance range: {min_distance*100:.1f}cm (closest) to {max_distance*100:.1f}cm (farthest)")
print(f"   Optimal range for Panda: 25-45cm")

if all_reachable:
    print("   All tray slots verified reachable!")
else:
    print("   WARNING: Some slots may be unreachable - adjust tray positions!")

# Reset robot to home position after verification
print("   Resetting robot to home position...")
home_joints = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]  # Safe home configuration
robot.control(home_joints)
sim_wait(200)

# ========== PICK AND PLACE ==========
print("\n" + "="*70)
print("STARTING INTELLIGENT SINGULATION")
print("="*70)
print("\n Strategy:")
print("   1. Detect vial size and position (CV simulation)")
print("   2. Open gripper to appropriate width")
print("   3. Approach and grasp gently")
print("   4. Create constraint for secure transport")
print("   5. Place in matching tray\n")

# Track placements
placed_counts = {'small': 0, 'medium': 0, 'large': 0}
vial_picked = [False] * len(vials)  # Track which vials have been picked
vial_skip_counts = [0] * len(vials)  # Track how many times each vial was skipped
vial_retry_counts = [0] * len(vials)  # Track retry attempts per vial (max 3)
MAX_RETRIES = 3  # Maximum retry attempts before skipping
current_constraint = None

def handle_grasp_failure(vial_idx, reason=""):
    """
    Handle grasp failure with retry logic.
    
    Returns:
        bool: True if should retry (haven't hit max retries), False if should skip
    """
    global vial_retry_counts, vial_skip_counts, consecutive_failures
    
    vial_retry_counts[vial_idx] += 1
    
    # CRITICAL: Reset robot to home position after failure
    # This prevents robot from getting stuck in weird configurations
    print(f"     Resetting robot to home position...")
    home_joints = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    robot.control(home_joints)
    sim_wait(100)  # Wait for robot to reach home
    
    if vial_retry_counts[vial_idx] < MAX_RETRIES:
        # Still have retries left
        print(f"       Grasp attempt {vial_retry_counts[vial_idx]}/{MAX_RETRIES} failed. {reason}")
        print(f"       Retrying vial {vial_idx + 1}...")
        return True  # Retry
    else:
        # Max retries reached, skip this vial
        print(f"      All {MAX_RETRIES} grasp attempts failed. {reason}")
        print(f"      Skipping vial {vial_idx + 1}")
        vial_skip_counts[vial_idx] += 1
        vial_retry_counts[vial_idx] = 0  # Reset for potential future attempts
        consecutive_failures += 1
        return False  # Skip

def recompute_isolation_scores(skip_counts):
    """Recompute isolation scores for all unpicked vials and visualize
    
    Args:
        skip_counts: List of skip counts for each vial
    """
    scores = []
    for i in range(len(vials)):
        if vial_picked[i]:
            continue
        
        # Base isolation score
        base_score = compute_isolation_score(i, vials)
        
        # Apply skip penalty: -10 points per skip
        skip_penalty = skip_counts[i] * 10.0
        final_score = max(0.0, base_score - skip_penalty)
        
        scores.append((i, final_score))
        
        # Visualize score
        vial_pos = get_position(vials[i])
        text_pos = [vial_pos[0], vial_pos[1], vial_pos[2] + 0.08]
        
        # Color based on score (after penalty)
        if final_score < 30:
            color = [1, 0, 0]  # Red - crowded or skipped
        elif final_score < 60:
            color = [1, 1, 0]  # Yellow - moderate
        else:
            color = [0, 1, 0]  # Green - isolated
        
        # Build display text
        if skip_counts[i] > 0:
            display_text = f"{final_score:.0f} (skipped x{skip_counts[i]})"
        else:
            display_text = f"{final_score:.0f}"
        
        p.addUserDebugText(
            text=display_text,
            textPosition=text_pos,
            textColorRGB=color,
            textSize=1.0,
            lifeTime=8.0  # Short lifetime, will be refreshed
        )
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def print_top_scores(scores, count=3, title="New top scores:"):
    """Print top N isolation scores"""
    if scores:
        print(f"     {title}")
        for i, (idx, score) in enumerate(scores[:count]):
            vt = vial_types[idx]
            skip_info = f" (skipped x{vial_skip_counts[idx]})" if vial_skip_counts[idx] > 0 else ""
            print(f"       #{i+1}: Vial {idx+1} ({vt.upper()}) - Score: {score:.1f}{skip_info}")

def test_placement_orientation(robot, target_pos, current_joints):
    """Test multiple orientations and find the best one WITHOUT actually moving robot
    
    Returns: (best_orientation, best_score, best_joints) or (None, 0, None)
    """
    # More orientation candidates for better coverage
    orientation_candidates = [
        ([np.pi, 0, 0], "vertical"),
        ([np.pi - 0.1, 0, 0], "tilt 6° forward"),
        ([np.pi + 0.1, 0, 0], "tilt 6° backward"),
        ([np.pi, 0.1, 0], "tilt 6° right"),
        ([np.pi, -0.1, 0], "tilt 6° left"),
        ([np.pi - 0.2, 0, 0], "tilt 11° forward"),
        ([np.pi + 0.2, 0, 0], "tilt 11° backward"),
        ([np.pi, 0.2, 0], "tilt 11° right"),
        ([np.pi, -0.2, 0], "tilt 11° left"),
        ([np.pi - 0.1, 0.1, 0], "tilt forward-right"),
        ([np.pi - 0.1, -0.1, 0], "tilt forward-left"),
    ]
    
    joint_limits = [
        (-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973),
        (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973)
    ]
    
    best_score = -10000
    best_orient = None
    best_q = None
    best_margin = -10
    
    for orient_angles, desc in orientation_candidates:
        test_orient = m.get_quaternion(orient_angles)
        
        # Test IK
        q_test = robot.ik(robot.end_effector, target_pos, test_orient, max_iterations=150)
        if q_test is None:
            continue
        
        # Calculate margins for all joints
        margins = []
        for angle, (min_lim, max_lim) in zip(q_test[:7], joint_limits):
            margin = min(angle - min_lim, max_lim - angle)
            margins.append(margin)
        
        min_margin = min(margins)
        
        # Allow slight violations (up to 2°) for borderline cases
        # Hard limit: reject if violation > 0.035 rad (~2°)
        if min_margin < -0.035:
            continue
        
        # Calculate smoothness (distance from current config)
        smoothness = np.linalg.norm(np.array(q_test[:7]) - np.array(current_joints[:7]))
        
        # Score: prioritize margin, but also consider smoothness
        # Penalize negative margins heavily
        margin_score = min_margin * 10.0 if min_margin >= 0 else min_margin * 50.0
        score = margin_score - smoothness * 0.05
        
        if score > best_score:
            best_score = score
            best_orient = test_orient
            best_q = q_test
            best_margin = min_margin
    
    # Diagnostic output
    if best_orient is None:
        print(f"     No valid orientation found (all candidates failed IK or exceeded limits)")
    elif best_margin < 0:
        print(f"     Selected orientation with minor limit violation (margin: {best_margin:.3f} rad = {np.degrees(best_margin):.1f}°)")
    
    return best_orient, best_score, best_q

# Initial computation
print("\nComputing initial isolation scores...")
isolation_scores = recompute_isolation_scores(vial_skip_counts)

print_top_scores(isolation_scores, count=5, title="Top 5 most isolated vials:")

# Visualize scores in 3D
print("Visualizing isolation scores above each vial...")
visualize_isolation_scores(vials, vial_skip_counts, label="Initial")

# Main picking loop - DYNAMIC isolation scoring
pick_cycle = 0
consecutive_failures = 0  # Track consecutive failures
AGGRESSIVE_MODE_THRESHOLD = 5  # Enter aggressive mode after 5 failures
SHAKE_BOX_THRESHOLD = 8  # Shake box after 8 consecutive failures
last_shake_failure_count = -999  # Track when we last shook

# Continue until all trays are full or no more graspable vials
max_attempts = len(vials) * 3  # Allow multiple passes through vials
attempt_count = 0

while attempt_count < max_attempts:
    attempt_count += 1
    
    # Check if all trays are full
    all_trays_full = all(
        placed_counts[vtype] >= VIAL_SPECS[vtype]['count'] 
        for vtype in ['small', 'medium', 'large']
    )
    if all_trays_full:
        print("\n All trays are full!")
        break
    
    # Recompute isolation scores for remaining vials
    isolation_scores = recompute_isolation_scores(vial_skip_counts)
    
    # Check if any vials remain
    remaining_vials = [(idx, score) for idx, score in isolation_scores if not vial_picked[idx]]
    if not remaining_vials:
        print("\n  No more vials to attempt")
        break
    
    # Get next best vial
    vial_idx, isolation_score = remaining_vials[0]
    vial = vials[vial_idx]
    vtype = vial_types[vial_idx]
    
    # SHAKE BOX: If too many consecutive failures, shake to redistribute
    if consecutive_failures >= SHAKE_BOX_THRESHOLD and consecutive_failures != last_shake_failure_count:
        print(f"\n SHAKE MODE: {consecutive_failures} consecutive failures")
        print(f"   Shaking box to redistribute vials...")
        if shake_box_with_gripper(robot, box_pieces, num_shakes=3, shake_distance=0.03):
            last_shake_failure_count = consecutive_failures
            # Recompute isolation scores after shake
            isolation_scores = recompute_isolation_scores(vial_skip_counts)
            print("   Isolation scores updated after shake")
        else:
            print("     Shake failed, continuing anyway...")
    
    # AGGRESSIVE MODE: If too many consecutive failures, lower standards
    if consecutive_failures >= AGGRESSIVE_MODE_THRESHOLD:
        print(f"\n  AGGRESSIVE MODE: {consecutive_failures} consecutive failures")
        print(f"   Lowering isolation threshold and trying harder...")
    spec = VIAL_SPECS[vtype]
    tray_config = TRAY_CONFIGS[vtype]
    
    # Highlight current target vial with larger text
    vial_pos = get_position(vial)
    highlight_text_id = p.addUserDebugText(
        text="TARGET",
        textPosition=[vial_pos[0], vial_pos[1], vial_pos[2] + 0.12],
        textColorRGB=[1, 1, 1],
        textSize=1.5,
        lifeTime=30.0
    )
    
    # Analyze and visualize optimal grasp for this vial
    print(f"  Analyzing optimal grasp configuration...")
    grasp_analysis = analyze_and_visualize_grasp(
        vial, vial_idx, vials, spec['height'], duration=30.0
    )
    print(f"    Tilt: {grasp_analysis['tilt_angle']:.1f} degrees")
    print(f"    Grasp type: {grasp_analysis['grasp_type']}")
    print(f"    Grasp height: {grasp_analysis['grasp_height_ratio']*100:.0f}% up vial")
    print(f"    Approach clearance: {grasp_analysis['clearance_score']:.0f}/100 "
          f"(min {grasp_analysis['min_clearance']*100:.1f}cm)")
    
    # Check if tray is full
    if is_tray_full(vtype, placed_counts[vtype]):
        print(f"\n  {vtype.upper()} tray full, skipping vial {vial_idx + 1}")
        vial_skip_counts[vial_idx] += 1
        # Recompute scores
        isolation_scores = recompute_isolation_scores(vial_skip_counts)
        print_top_scores(isolation_scores, count=3, title="After tray full:")
        continue
    
    print(f"\n{'='*60}")
    print(f"VIAL {vial_idx + 1}/{len(vials)} - {vtype.upper()} "
          f"({spec['radius']*1000:.0f}mm, {spec['mass']*1000:.0f}g) "
          f"[Isolation Score: {isolation_score:.1f}]")
    print('='*60)
    
    try:
        # REAL-TIME CV detection
        vial_pos = get_position(vial)
        vial_orient = get_orientation(vial)
        vial_axis = get_vial_axis(vial_orient)
        
        print(f" Position: [{vial_pos[0]:.3f}, {vial_pos[1]:.3f}, {vial_pos[2]:.3f}]")
        print(f" Axis: [{vial_axis[0]:.2f}, {vial_axis[1]:.2f}, {vial_axis[2]:.2f}]")
        
        # Check isolation (adaptive threshold based on mode)
        if consecutive_failures >= AGGRESSIVE_MODE_THRESHOLD:
            isolation_threshold = 0.03  # 3cm in aggressive mode
        else:
            isolation_threshold = 0.04  # 4cm normal
        
        is_isolated, nearest_dist, nearest_idx, crowded_count = check_grasp_isolation(
            vial_pos, vials, vial_idx, min_distance=isolation_threshold
        )
        
        print(f" Isolation: {nearest_dist*100:.1f}cm to nearest vial")
        
        # If not isolated, try singulation push
        if not is_isolated:
            print(f"  Vial too close to neighbors ({crowded_count} within 4cm)")
            print(f"  Attempting singulation push...")
            
            # Record positions before push
            nearest_vial_pos_before = get_position(vials[nearest_idx])
            target_vial_pos_before = vial_pos.copy()
            
            push_traj = calculate_push_trajectory(vial_pos, nearest_vial_pos_before, push_distance=0.08)
            
            if push_traj:
                push_start, push_end, push_orient = push_traj
                push_distance_planned = np.linalg.norm(np.array(push_end[:2]) - np.array(push_start[:2]))
                print(f"     Push trajectory: {push_distance_planned*100:.1f}cm")
                print(f"     Start: [{push_start[0]:.3f}, {push_start[1]:.3f}]")
                print(f"     End:   [{push_end[0]:.3f}, {push_end[1]:.3f}]")
                
                # Visualize push trajectory
                p.addUserDebugLine(push_start, push_end, [1, 0.5, 0], lineWidth=4, lifeTime=15)  # Orange push line
                
                if singulation_push(robot, push_start, push_end, push_orient, steps=500):
                    print(f"  Waiting for physics to settle...")
                    sim_wait(500)
                    
                    # Measure actual movement
                    nearest_vial_pos_after = get_position(vials[nearest_idx])
                    target_vial_pos_after = get_position(vial)
                    
                    nearest_moved = np.linalg.norm(np.array(nearest_vial_pos_after[:2]) - np.array(nearest_vial_pos_before[:2]))
                    target_moved = np.linalg.norm(np.array(target_vial_pos_after[:2]) - np.array(target_vial_pos_before[:2]))
                    
                    print(f"     Nearest vial moved: {nearest_moved*100:.1f}cm")
                    print(f"     Target vial moved: {target_moved*100:.1f}cm")
                    
                    # Recompute isolation scores after push (positions changed)
                    print(f"  Recomputing isolation scores after push...")
                    isolation_scores = recompute_isolation_scores(vial_skip_counts)
                    print_top_scores(isolation_scores, count=3, title="After push:")
                    
                    # Update visualization
                    visualize_isolation_scores(vials, vial_skip_counts, label="After push")
                    
                    # Re-check isolation (lowered threshold)
                    vial_pos = target_vial_pos_after  # Update position
                    is_isolated, nearest_dist, nearest_idx, crowded_count = check_grasp_isolation(
                        vial_pos, vials, vial_idx, min_distance=0.04
                    )
                    print(f" After push: {nearest_dist*100:.1f}cm to nearest")
                    
                    # Warn if push was ineffective
                    if nearest_moved < 0.01 and target_moved < 0.01:
                        print(f"     WARNING: Push had no effect! Vials didn't move.")
                    
                    # CHANGED: Don't skip immediately if still crowded
                    # Instead, try to generate grasp candidates - might still work!
                    if not is_isolated:
                        print(f"  Still crowded after push ({nearest_dist*100:.1f}cm), but will try grasp anyway...")
                        # Continue to grasp generation (don't skip yet)
                else:
                    print(f" Push failed, skipping...")
                    vial_skip_counts[vial_idx] += 1
                    isolation_scores = recompute_isolation_scores(vial_skip_counts)
                    print_top_scores(isolation_scores, count=3, title="After push failed:")
                    continue
            else:
                print(f" Cannot calculate push trajectory, skipping...")
                vial_skip_counts[vial_idx] += 1
                isolation_scores = recompute_isolation_scores(vial_skip_counts)
                print_top_scores(isolation_scores, count=3, title="Cannot push:")
                continue
        
        # ========== GRASP COMPUTATION ==========
        if USE_MULTI_CANDIDATE_GRASPS:
            # MULTI-CANDIDATE: Generate and score multiple grasps
            print(f"  Generating {NUM_GRASP_ROTATIONS}x{NUM_GRASP_HEIGHTS}x{NUM_GRASP_APPROACHES} grasp candidates...")
            
            candidates = generate_grasp_candidates(
                vial_pos, vial_orient,
                spec['radius'], spec['height'],
                num_rotations=NUM_GRASP_ROTATIONS,
                num_heights=NUM_GRASP_HEIGHTS,
                num_approaches=NUM_GRASP_APPROACHES,
                vial_idx=vial_idx
            )
            
            best_grasp = select_best_grasp(
                candidates, robot, vial,
                spec['mass'], spec['radius'],
                mu=VIAL_FRICTION,
                top_k=5, verbose=True,
                all_vials=vials, vial_idx=vial_idx
            )
            
            if best_grasp is None:
                print(f"  No valid grasp found!")
                if not handle_grasp_failure(vial_idx, "No valid grasp candidate"):
                    # Max retries reached, skip
                    isolation_scores = recompute_isolation_scores(vial_skip_counts)
                    print_top_scores(isolation_scores, count=3, title="No grasp:")
                continue  # Retry or skip
            
            # Extract grasp parameters
            grasp_pos = best_grasp.position.copy()  # Use candidate position directly
            gripper_orient = best_grasp.orientation
            approach_dir = np.array([0, 0, 1])  # Always from above
            grasp_type = f"Multi-candidate (score={best_grasp.total_score:.2f})"
            
            # Debug: print grasp position
            print(f"  Best grasp position: [{grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f}]")
            
        else:
            # SIMPLE: Single grasp computation (original method)
            grasp_pos, gripper_orient, approach_dir, grasp_type = compute_grasp_orientation(
                vial_pos, vial_orient
            )
        
        print(f"Grasp: {grasp_type}")
        
        # Visualize grasp orientation
        if USE_MULTI_CANDIDATE_GRASPS:
            from scipy.spatial.transform import Rotation as R
            R_gripper = R.from_quat(gripper_orient)
            gripper_z = R_gripper.apply([0, 0, 1])  # Approach direction
            gripper_y = R_gripper.apply([0, 1, 0])  # Finger direction
            
            # Draw approach direction (blue arrow)
            p.addUserDebugLine(grasp_pos, grasp_pos + gripper_z * 0.10, 
                             [0, 0, 1], lineWidth=3, lifeTime=10)
            # Draw finger direction (green arrow)
            p.addUserDebugLine(grasp_pos, grasp_pos + gripper_y * 0.08, 
                             [0, 1, 0], lineWidth=3, lifeTime=10)
            
            print(f"  Grasp orientation: approach=[{gripper_z[0]:.2f}, {gripper_z[1]:.2f}, {gripper_z[2]:.2f}], "
                  f"fingers=[{gripper_y[0]:.2f}, {gripper_y[1]:.2f}, {gripper_y[2]:.2f}]")
        
        # Target slot
        row = placed_counts[vtype] // tray_config['cols']
        col = placed_counts[vtype] % tray_config['cols']
        target_pos = compute_target_slot_position(vtype, row, col)
        
        print(f"Target: {vtype.upper()} tray ({row},{col}) at "
              f"[{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        
        # 1. Approach
        print(f"  Approaching...")
        vial_pos = get_position(vial)  # Update
        above_pos = vial_pos + approach_dir * 0.25
        if not move_to_pose(robot, above_pos, gripper_orient, steps=600):
            print("     Approach failed")
            if not handle_grasp_failure(vial_idx, "IK/approach failed"):
                # Max retries reached, skip
                isolation_scores = recompute_isolation_scores(vial_skip_counts)
                print_top_scores(isolation_scores, count=3, title="Approach failed:")
            continue  # Retry or skip
        
        # 2. Open to appropriate size (wider for horizontal vials)
        # Check if vial is horizontal
        vial_orient_check = get_orientation(vial)
        vial_axis_check = get_vial_axis(vial_orient_check)
        tilt_check = np.degrees(np.arccos(np.clip(abs(np.dot(vial_axis_check, [0,0,1])), 0, 1)))
        is_horizontal = tilt_check > 60
        
        if is_horizontal:
            print(f" Opening gripper WIDE for horizontal vial: {spec['radius']*4*1000:.1f}mm...")
        else:
            print(f" Opening gripper to {spec['radius']*3*1000:.1f}mm...")
        
        open_gripper(robot, spec['radius'], is_horizontal=is_horizontal)
        sim_wait(150)
        
        # 3. Move to grasp (descend slowly)
        print("  Descending to grasp...")
        
        # Update vial position for real-time CV tracking
        vial_pos_updated = get_position(vial)
        
        # Update grasp position based on vial movement (keep orientation from best_grasp)
        if USE_MULTI_CANDIDATE_GRASPS:
            # Multi-candidate: Keep the carefully computed orientation, only update position
            position_delta = vial_pos_updated - vial_pos
            grasp_pos = grasp_pos + position_delta  # Adjust for vial movement
            # gripper_orient and approach_dir already set from best_grasp
        else:
            # Simple: Recompute everything
            vial_orient = get_orientation(vial)
            grasp_pos, gripper_orient, approach_dir, _ = compute_grasp_orientation(
                    vial_pos_updated, vial_orient
            )
        
        # Descend in steps to avoid collision, approach from above grasp_pos
        # Enable collision checking during descent (check box/table, NOT vials)
        descent_success = True
        ik_failed = False
        for desc_frac in [0.7, 0.4, 0.1, 0.0]:
            intermediate_pos = grasp_pos + approach_dir * (0.25 * desc_frac)
            if not move_to_pose(robot, intermediate_pos, gripper_orient, steps=200,
                              check_collisions=ENABLE_COLLISION_CHECK, 
                              box_pieces=box_pieces, table=table, held_vial=None):
                # Check if it was IK failure or collision
                # (IK failure prints "[IK FAILED]" message)
                descent_success = False
                break
        
            # VERIFY: Check if gripper is actually inside box
            ee_state = p.getLinkState(robot.body, robot.end_effector)
            ee_pos_actual = np.array(ee_state[0])
            box_center = np.array(BOX_POS)
            box_half = np.array(BOX_SIZE) / 2
            
            # Check if gripper is outside box (with margin for gripper width)
            # Margin accounts for gripper fingers (~4cm radius) + safety (1cm)
            GRIPPER_MARGIN = 0.05  # 5cm margin (was 0.10)
            outside_x = abs(ee_pos_actual[0] - box_center[0]) > box_half[0] - GRIPPER_MARGIN
            outside_y = abs(ee_pos_actual[1] - box_center[1]) > box_half[1] - GRIPPER_MARGIN
            
            if outside_x or outside_y:
                print(f"     WARNING: Gripper outside box at [{ee_pos_actual[0]:.3f}, {ee_pos_actual[1]:.3f}, {ee_pos_actual[2]:.3f}]")
                print(f"     Box bounds: X=[{box_center[0]-box_half[0]:.3f}, {box_center[0]+box_half[0]:.3f}], "
                      f"Y=[{box_center[1]-box_half[1]:.3f}, {box_center[1]+box_half[1]:.3f}]")
                descent_success = False
                break
        
        if not descent_success:
            open_gripper(robot, spec['radius'])
            
            # IK failure or collision - both count as grasp failure
            if not handle_grasp_failure(vial_idx, "IK/Collision during descent"):
                # Max retries reached, skip
                print(f"  Recomputing isolation scores after descent failure...")
                isolation_scores = recompute_isolation_scores(vial_skip_counts)
                print_top_scores(isolation_scores, count=3, title="After descent fail:")
                visualize_isolation_scores(vials, vial_skip_counts, label="After descent fail")
            continue  # Retry or skip
        
        # 3.5. FINE ADJUSTMENT - Disabled for now
        # The grasp candidate scoring should already select good positions
        # Fine adjustment was causing issues with large offsets
        # If needed in future, implement XY adjustment only (not Z)
        
        # 4. Close to grip (different strategy for horizontal vials)
        # Calculate vial tilt
        vial_axis = get_vial_axis(vial_orient)
        vertical = np.array([0, 0, 1])
        tilt_angle = np.arccos(np.clip(abs(np.dot(vial_axis, vertical)), 0, 1))
        tilt_degrees = np.degrees(tilt_angle)
        
        # For horizontal vials (>60°), need DIFFERENT closing strategy
        if tilt_degrees > 60:
            # FIXED: Horizontal vials - close to 2.0x diameter (not 2.8x)
            # 2.0x ensures fingers make solid contact without being too loose
            close_amount = spec['radius'] * 2.0  # 1.0 * diameter
            grip_force = GRIPPER_FORCE * 1.5  # 50% more force
            grip_steps = 600  # Even more time
            print(f" Closing for horizontal vial (tilt={tilt_degrees:.0f}°, "
                  f"target={close_amount*1000:.1f}mm, force={grip_force:.0f}N)...")
        else:
            # Vertical: normal side grasp
            close_multiplier = GRIPPER_CLOSE_MULTIPLIER
            close_amount = spec['radius'] * close_multiplier
            grip_force = GRIPPER_FORCE
            grip_steps = 400
            print(f" Closing to grip (target: {close_amount*1000:.1f}mm, "
                  f"force={grip_force:.0f}N)...")
        
        robot.set_gripper_position([close_amount, close_amount], 
                                   set_instantly=False, 
                                   force=grip_force)
        sim_wait(grip_steps)
        
        # Check initial grip - IMPROVED validation
        initial_pos = get_position(vial)
        moved_initial = np.linalg.norm(initial_pos - vial_pos)
        print(f" Initial grip moved vial: {moved_initial*1000:.1f}mm")
        
        # Additional check: gripper finger positions
        # Get actual gripper joint angles (positions)
        gripper_angles = robot.get_joint_angles(joints=robot.gripper_joints)
        actual_gripper_width = gripper_angles[0] + gripper_angles[1]
        expected_width = close_amount * 2
        gripper_closed_properly = abs(actual_gripper_width - expected_width) < 0.005  # 5mm tolerance
        
        # Grip is valid if:
        # 1. Vial moved (indicates contact), OR
        # 2. Vial moved very little BUT gripper closed to expected position (firm grip)
        grip_valid = moved_initial >= MIN_GRIP_MOVEMENT or \
                     (moved_initial >= 0.0001 and gripper_closed_properly)
        
        if not grip_valid:
            print(f"     Vial not gripped (moved={moved_initial*1000:.1f}mm, "
                  f"gripper={actual_gripper_width*1000:.1f}mm vs expected={expected_width*1000:.1f}mm)")
            open_gripper(robot, spec['radius'])
            
            if not handle_grasp_failure(vial_idx, "Gripper didn't grip vial"):
                # Max retries reached, skip
                print(f"  Recomputing isolation scores after skip...")
                isolation_scores = recompute_isolation_scores(vial_skip_counts)
                print_top_scores(isolation_scores, count=3, title="After skip:")
                visualize_isolation_scores(vials, vial_skip_counts, label="After skip")
            continue  # Retry or skip
        
        # 5. Create constraint
        print("  Creating constraint...")
        current_constraint = create_vial_constraint(robot, vial)
        
        if current_constraint is not None:
            print("     Vial attached via constraint!")
            
            # STABILIZATION: Let constraint settle to reduce shaking
            # Apply damping to vial motion
            vial_body_id = vial if isinstance(vial, int) else vial.body
            p.changeDynamics(vial_body_id, -1,
                           linearDamping=2.0,  # Increased damping during transport
                           angularDamping=2.0)
        else:
            print("     Constraint failed, using friction only")
        
        # IMPORTANT: Keep gripper closed during stabilization
        # The gripper needs continuous force to maintain grip
        for i in range(300):  # Longer stabilization period
            # Re-apply closing force every 50 steps
            if i % 50 == 0:
                robot.set_gripper_position([close_amount, close_amount], 
                                          set_instantly=False, 
                                          force=GRIPPER_FORCE)
            m.step_simulation()
        
        # 6. Lift slowly while gradually re-orienting to vertical
        print("  Lifting and re-orienting to vertical...")
        lift_success = True
        base_vial_pos = get_position(vial)  # Record starting position
        initial_vial_orient = get_orientation(vial)
        initial_vial_axis = get_vial_axis(initial_vial_orient)
        initial_vial_tilt = np.degrees(np.arccos(np.clip(abs(initial_vial_axis[2]), 0, 1)))
        
        print(f"     Initial vial tilt: {initial_vial_tilt:.1f}°")
        
        # Use proper vertical placement calculation
        # This accounts for the relative offset between gripper and vial
        for lift_step in range(3):
            lift_height = 0.10 * (lift_step + 1)  # 10cm increments (faster)
            
            # Target vial position at this height
            target_vial_pos = initial_pos + approach_dir * lift_height
            
            # Calculate gripper pose to make vial vertical at target
            lift_pos, lift_orient = calculate_vertical_placement_pose(
                robot, vial, target_vial_pos
            )
            
            move_to_pose(robot, lift_pos, lift_orient, steps=300)
            
            # MAINTAIN GRIP during lifting
            robot.set_gripper_position([close_amount, close_amount], 
                                      set_instantly=False, 
                                      force=GRIPPER_FORCE)
            
            # Check if vial followed
            current_vial_pos = get_position(vial)
            vial_lifted = current_vial_pos[2] - base_vial_pos[2]
            
            # Check vial orientation change
            current_vial_orient = get_orientation(vial)
            current_vial_axis = get_vial_axis(current_vial_orient)
            current_vial_tilt = np.degrees(np.arccos(np.clip(abs(current_vial_axis[2]), 0, 1)))
            
            if lift_step > 0 and vial_lifted < MIN_LIFT_HEIGHT:
                print(f"     Vial not lifting! (only {vial_lifted*100:.1f}cm at step {lift_step})")
                lift_success = False
                if current_constraint:
                    remove_constraint(current_constraint)
                    current_constraint = None
                break
            
            if lift_step % 2 == 0:
                print(f"     Lifted {vial_lifted*100:.1f}cm, tilt now {current_vial_tilt:.1f}° (target: 0°)")
        
        if not lift_success:
            open_gripper(robot, spec['radius'])
            vial_skip_counts[vial_idx] += 1
            isolation_scores = recompute_isolation_scores(vial_skip_counts)
            print_top_scores(isolation_scores, count=3, title="Lift failed:")
            continue
        
        # 6.5. FORCE vial to vertical orientation before placement
        # Apply direct orientation correction WITHOUT removing constraint
        if current_vial_tilt > 15:  # If still tilted more than 15°
            print(f"  Forcing vial to vertical (current tilt: {current_vial_tilt:.1f}°)...")
            vial_body_id = vial if isinstance(vial, int) else vial.body
            current_vial_pos = get_position(vial)
            vertical_orient = m.get_quaternion([np.pi, 0, 0])  # Vertical orientation
            
            # Get current end effector state
            ee_state = p.getLinkState(robot.body, robot.end_effector)
            ee_pos = np.array(ee_state[0])
            ee_orient = np.array(ee_state[1])
            
            # Calculate new relative position (keep vial at current position)
            rel_pos = current_vial_pos - ee_pos
            
            # Calculate new relative orientation for VERTICAL vial
            from scipy.spatial.transform import Rotation as R
            ee_rot = R.from_quat(ee_orient)
            vial_rot = R.from_quat(vertical_orient)
            rel_rot = ee_rot.inv() * vial_rot
            rel_orient = rel_rot.as_quat()
            
            # Update constraint to lock orientation to VERTICAL
            if current_constraint:
                # Change constraint type to FIXED with vertical orientation
                remove_constraint(current_constraint)
                
                # Create new FIXED constraint with vertical orientation locked
                current_constraint = p.createConstraint(
                    parentBodyUniqueId=robot.body,
                    parentLinkIndex=robot.end_effector,
                    childBodyUniqueId=vial_body_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,  # Lock BOTH position and orientation
                    jointAxis=[0, 0, 0],
                    parentFramePosition=rel_pos.tolist(),
                    childFramePosition=[0, 0, 0],
                    parentFrameOrientation=rel_orient.tolist(),  # Lock to vertical!
                    childFrameOrientation=[0, 0, 0, 1]
                )
            
            # Wait for physics to settle
            sim_wait(50)
            
            # Verify orientation
            final_vial_orient = get_orientation(vial)
            final_vial_axis = get_vial_axis(final_vial_orient)
            final_vial_tilt = np.degrees(np.arccos(np.clip(abs(final_vial_axis[2]), 0, 1)))
            print(f"     Vial orientation locked to vertical, tilt now: {final_vial_tilt:.1f}°")
        
        # 7. Move to tray with better base joint configuration
        print(f"  Moving to {vtype.upper()} tray...")
        above_target = [target_pos[0], target_pos[1], target_pos[2] + 0.30]
        
        # Target orientation: vertical down (for placement)
        final_orient = m.get_quaternion([np.pi, 0, 0])
        
        # Calculate optimal base rotation (joint 1) to face tray
        dx = target_pos[0] - 0.0  # Tray X - Robot base X
        dy = target_pos[1] - 0.0  # Tray Y - Robot base Y  
        optimal_base_angle = np.arctan2(dy, dx)  # Angle to face tray
        
        print(f"     Trying multiple base rotations to find best arm configuration...")
        print(f"     Target tray direction: {np.degrees(optimal_base_angle):.0f}° from robot")
        
        # Try IK with different base joint angles to avoid arm blocking
        q_target = None
        best_config = None
        best_margin = -1000
        
        # Save current state
        saved_joints = robot.get_joint_angles()
        
        for attempt, base_offset in enumerate([0, np.pi, -np.pi/2, np.pi/2, np.pi*3/4, -np.pi*3/4]):
            test_angle = optimal_base_angle + base_offset
            
            # Seed robot with this base angle
            test_joints = saved_joints.copy()
            test_joints[0] = test_angle
            robot.control(test_joints[:7])
            
            # Small simulation to update robot state
            robot.set_gripper_position([close_amount, close_amount], force=GRIPPER_FORCE)
            sim_wait(10)
            
            # Try IK from this configuration
            q_test = robot.ik(robot.end_effector, above_target, final_orient, max_iterations=150)
            
            if q_test is not None:
                # Check joint margins
                joint_limits = [
                    (-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973),
                    (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973)
                ]
                
                margins = [min(angle - lim[0], lim[1] - angle) for angle, lim in zip(q_test[:7], joint_limits)]
                min_margin = min(margins)
                
                if min_margin > best_margin:
                    best_margin = min_margin
                    q_target = q_test
                    best_config = attempt + 1
                    
                # Show result
                status = "GOOD" if min_margin > 0.1 else "RISKY" if min_margin > 0 else "BAD"
                print(f"       Attempt {attempt+1}: base={np.degrees(q_test[0]):.0f}°, margin={min_margin:.3f} rad [{status}]")
                
                # If excellent, use it
                if min_margin > 0.15:  # 0.15 rad = 8.6° margin
                    break
        
        if q_target is None:
            print(f"     ERROR: Cannot reach above_target with any base rotation!")
            robot.control(saved_joints[:7])  # Restore
            if current_constraint:
                remove_constraint(current_constraint)
                current_constraint = None
            open_gripper(robot, spec['radius'])
            vial_skip_counts[vial_idx] += 1
            isolation_scores = recompute_isolation_scores(vial_skip_counts)
            print_top_scores(isolation_scores, count=3, title="Cannot reach tray:")
            continue
        else:
            print(f"     Using configuration #{best_config}: base at {np.degrees(q_target[0]):.0f}°, margin={best_margin:.3f} rad")
        
        # Move to above target with chosen configuration
        for step in range(1200):
            robot.control(q_target[:7])
            robot.set_gripper_position([close_amount, close_amount], 
                                      set_instantly=False, 
                                      force=GRIPPER_FORCE)
            m.step_simulation()
        
        # MAINTAIN GRIP during transport
        robot.set_gripper_position([close_amount, close_amount], 
                                  set_instantly=False, 
                                  force=GRIPPER_FORCE)
        
        # Wait for stabilization at above_target position
        sim_wait(100)
        
        # Verify we reached above_target
        ee_pos_above, _ = robot.get_link_pos_orient(robot.end_effector)
        ee_to_above_dist = np.linalg.norm(np.array(ee_pos_above[:2]) - np.array(above_target[:2]))
        
        print(f"     End effector to above_target: {ee_to_above_dist*1000:.1f}mm")
        
        if ee_to_above_dist > 0.05:  # More than 5cm off
            print(f"     ERROR: Cannot reach above_target!")
            print(f"     Target: [{above_target[0]:.3f}, {above_target[1]:.3f}, {above_target[2]:.3f}]")
            print(f"     EE pos: [{ee_pos_above[0]:.3f}, {ee_pos_above[1]:.3f}, {ee_pos_above[2]:.3f}]")
            
            # Remove constraint and skip
            if current_constraint:
                remove_constraint(current_constraint)
                current_constraint = None
            open_gripper(robot, spec['radius'])
            vial_skip_counts[vial_idx] += 1
            isolation_scores = recompute_isolation_scores(vial_skip_counts)
            print_top_scores(isolation_scores, count=3, title="Cannot reach tray:")
            continue
        
        # Check if still holding
        current_vial_pos = get_position(vial)
        dist_to_target = np.linalg.norm(current_vial_pos[:2] - np.array(target_pos[:2]))
        vial_height = current_vial_pos[2]
        
        print(f"     Vial XY distance to target: {dist_to_target*1000:.1f}mm")
        print(f"     Vial height: {vial_height:.3f}m (should be >0.9m)")
        
        if dist_to_target > 0.5 or vial_height < 0.85:
            print(f"     Vial lost during transport!")
            if current_constraint:
                remove_constraint(current_constraint)
                current_constraint = None
            vial_skip_counts[vial_idx] += 1
            isolation_scores = recompute_isolation_scores(vial_skip_counts)
            print_top_scores(isolation_scores, count=3, title="Vial lost:")
            continue
        
        # 8. SELF-CALIBRATION: Measure offset right before descending
        # This gives the most accurate compensation for the actual placement
        print("  Self-calibrating vial position...")
        
        ee_pos_calibrate, _ = robot.get_link_pos_orient(robot.end_effector)
        vial_pos_calibrate = get_position(vial)
        vial_offset_xy = np.array(vial_pos_calibrate[:2]) - np.array(ee_pos_calibrate[:2])
        
        print(f"     Measured offset: [{vial_offset_xy[0]*1000:.1f}, {vial_offset_xy[1]*1000:.1f}]mm")
        
        # 9. Descend with compensated target
        print("  Descending to slot...")
        
        # Compensate for offset in target position
        adjusted_target = [
            target_pos[0] - vial_offset_xy[0],  # Compensate X
            target_pos[1] - vial_offset_xy[1],  # Compensate Y
            target_pos[2]
        ]
        
        print(f"     Original target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        print(f"     Adjusted target: [{adjusted_target[0]:.3f}, {adjusted_target[1]:.3f}, {adjusted_target[2]:.3f}]")
        print(f"     Compensation: [{-vial_offset_xy[0]*1000:.1f}, {-vial_offset_xy[1]*1000:.1f}]mm")
        
        # CRITICAL: Use direct orientation control during descent
        # Constraint-based approach is unstable, use direct reset instead
        descent_orient = final_orient  # Use the same orientation throughout descent
        vertical_vial_orient = m.get_quaternion([np.pi, 0, 0])  # Target vial orientation (vertical)
        
        # Descend in stages to adjusted target
        descend_success = True
        for desc_step in range(5):
            desc_height = 0.30 - 0.06 * (desc_step + 1)
            desc_pos = [adjusted_target[0], adjusted_target[1], adjusted_target[2] + desc_height]
            
            # Use SAME orientation for all descent steps
            if not move_to_pose(robot, desc_pos, descent_orient, steps=250):
                print(f"     WARNING: Descend step {desc_step+1} may have failed (IK issue)")
                descend_success = False
            
            # MAINTAIN GRIP during descending
            robot.set_gripper_position([close_amount, close_amount], 
                                      set_instantly=False, 
                                      force=GRIPPER_FORCE)
        
            # FORCE vial to stay vertical using direct orientation reset
            # This is more reliable than constraint re-locking
            check_vial_orient = get_orientation(vial)
            check_vial_axis = get_vial_axis(check_vial_orient)
            check_vial_tilt = np.degrees(np.arccos(np.clip(abs(check_vial_axis[2]), 0, 1)))
            
            if check_vial_tilt > 15:  # If vial has tilted more than 15°
                print(f"     🔧 Correcting vial orientation (tilt: {check_vial_tilt:.1f}°)")
                
                # Get current vial position
                current_vial_pos = get_position(vial)
                
                # Directly reset vial to vertical orientation at current position
                # This is more stable than re-creating constraints
                p.resetBasePositionAndOrientation(
                    vial_body_id,
                    current_vial_pos,
                    vertical_vial_orient
                )
                
                # Give physics a moment to settle
                for _ in range(10):
                    robot.set_gripper_position([close_amount, close_amount], 
                                              set_instantly=False, 
                                              force=GRIPPER_FORCE)
                    m.step_simulation()
        
        # Final descent to target with SAME orientation
        if not move_to_pose(robot, adjusted_target, descent_orient, steps=300):
            print(f"     WARNING: Final descend to target may have failed")
            descend_success = False
        
        # FINAL orientation correction before release
        final_check_orient = get_orientation(vial)
        final_check_axis = get_vial_axis(final_check_orient)
        final_check_tilt = np.degrees(np.arccos(np.clip(abs(final_check_axis[2]), 0, 1)))
        if final_check_tilt > 15:
            print(f"     🔧 Final correction (tilt: {final_check_tilt:.1f}°)")
            
            # Final direct reset to vertical
            current_vial_pos = get_position(vial)
            p.resetBasePositionAndOrientation(
                vial_body_id,
                current_vial_pos,
                vertical_vial_orient
            )
            
            # Settle
            for _ in range(20):
                robot.set_gripper_position([close_amount, close_amount], 
                                          set_instantly=False, 
                                          force=GRIPPER_FORCE)
                m.step_simulation()
        
        # Verify actual end effector position
        ee_pos_final, _ = robot.get_link_pos_orient(robot.end_effector)
        ee_to_target_dist = np.linalg.norm(np.array(ee_pos_final[:2]) - np.array(adjusted_target[:2]))
        
        # DIAGNOSTIC: Check joint limits during descend
        current_joints = robot.get_joint_angles()[:7]
        joint_limits = [
            (-2.8973, 2.8973),  # Joint 1
            (-1.7628, 1.7628),  # Joint 2
            (-2.8973, 2.8973),  # Joint 3
            (-3.0718, -0.0698), # Joint 4
            (-2.8973, 2.8973),  # Joint 5
            (-0.0175, 3.7525),  # Joint 6
            (-2.8973, 2.8973),  # Joint 7
        ]
        
        joints_at_descend = []
        for i, (angle, (min_lim, max_lim)) in enumerate(zip(current_joints, joint_limits)):
            margin_min = angle - min_lim
            margin_max = max_lim - angle
            min_margin = min(margin_min, margin_max)
            if min_margin < 0.2:  # Within 0.2 rad of limit
                joints_at_descend.append((i+1, angle, min_lim, max_lim, min_margin))
        
        print(f"     End effector to target: {ee_to_target_dist*1000:.1f}mm")
        if joints_at_descend:
            print(f"     WARNING: Joints near limits at descend:")
            for joint_num, angle, min_l, max_l, margin in joints_at_descend:
                print(f"       Joint {joint_num}: {angle:.3f} rad (limits: [{min_l:.3f}, {max_l:.3f}], margin: {margin:.3f} rad)")
        
        # Accept if within slot tolerance (slot is 6.5-7cm, so 3cm margin is reasonable)
        slot_size = tray_config['slot_size']
        acceptable_error = slot_size * 0.4  # 40% of slot size
        
        if ee_to_target_dist > acceptable_error:
            print(f"     WARNING: End effector {ee_to_target_dist*1000:.1f}mm from target (limit: {acceptable_error*1000:.1f}mm)")
            print(f"     Target: [{adjusted_target[0]:.3f}, {adjusted_target[1]:.3f}, {adjusted_target[2]:.3f}]")
            print(f"     EE pos: [{ee_pos_final[0]:.3f}, {ee_pos_final[1]:.3f}, {ee_pos_final[2]:.3f}]")
            
            # Try corrective move: use current EE position as target
            # This often helps when IK is at workspace limit
            print(f"     Attempting corrective move using vial position...")
            vial_pos_now = get_position(vial)
            corrected_target = [
                vial_pos_now[0],
                vial_pos_now[1],
                target_pos[2]  # Same Z as slot
            ]
            
            print(f"     Corrected target: [{corrected_target[0]:.3f}, {corrected_target[1]:.3f}, {corrected_target[2]:.3f}]")
            
            # Try to move to corrected position
            move_to_pose(robot, corrected_target, final_orient, steps=200)
            sim_wait(100)
            
            # Re-verify
            vial_pos_after = get_position(vial)
            dist_after = np.linalg.norm(vial_pos_after[:2] - np.array(target_pos[:2]))
            print(f"     After correction: vial {dist_after*1000:.1f}mm from slot center")
            
            if dist_after > slot_size * 0.6:  # Still too far (more than 60% of slot)
                print(f"     Correction failed, skipping...")
                if current_constraint:
                    remove_constraint(current_constraint)
                    current_constraint = None
                open_gripper(robot, spec['radius'])
                vial_skip_counts[vial_idx] += 1
                isolation_scores = recompute_isolation_scores(vial_skip_counts)
                print_top_scores(isolation_scores, count=3, title="After IK failure:")
                continue
            else:
                print(f"     Correction successful! Proceeding with placement...")
        
        # Final grip reinforcement before release
        robot.set_gripper_position([close_amount, close_amount], 
                                  set_instantly=False, 
                                  force=GRIPPER_FORCE)
        
        # 10. Release (let physics handle vial orientation naturally)
        print("  Releasing...")
        
        vial_body_id = vial if isinstance(vial, int) else vial.body
        
        # Reset velocity to prevent flying away
        p.resetBaseVelocity(vial_body_id, linearVelocity=[0,0,0], angularVelocity=[0,0,0])
        
        # Restore normal damping for placement
        p.changeDynamics(vial_body_id, -1,
                       linearDamping=LINEAR_DAMPING,
                       angularDamping=ANGULAR_DAMPING)
        
        # Small wait to stabilize
        sim_wait(50)
        
        # Remove constraint - vial will drop into slot
        if current_constraint:
            remove_constraint(current_constraint)
            current_constraint = None
        
        # Wait before opening gripper (vial drops slightly into slot)
        sim_wait(100)
        
        print("  Opening gripper...")
        open_gripper(robot, spec['radius'])
        
        # Wait for vial to settle in slot naturally
        sim_wait(400)
        
        # Check vial orientation after settling
        final_vial_orient = get_orientation(vial)
        final_vial_axis = get_vial_axis(final_vial_orient)
        final_tilt = np.degrees(np.arccos(np.clip(abs(final_vial_axis[2]), 0, 1)))
        if final_tilt < 30:
            print(f"     Vial settled vertically (tilt: {final_tilt:.1f}°)")
        else:
            print(f"     Vial settled tilted (tilt: {final_tilt:.1f}°) - may affect check")
        
        # 11. Retreat
        print("  Retreating...")
        move_to_pose(robot, above_target, final_orient, steps=350)
        
        # Final check
        if check_placement_success(vial, target_pos):
            print(f"  SUCCESS! {vtype.upper()} vial placed!")
            consecutive_failures = 0  # Reset on success
            vial_retry_counts[vial_idx] = 0  # Reset retry counter on success
            placed_counts[vtype] += 1
            vial_picked[vial_idx] = True  # Mark as picked
            
            # Update debug text to show DONE
            final_pos = get_position(vial)
            p.addUserDebugText(
                text="DONE",
                textPosition=[final_pos[0], final_pos[1], final_pos[2] + 0.08],
                textColorRGB=[0, 0, 1],  # Blue
                textSize=1.0,
                lifeTime=60.0
            )
            
            # RECOMPUTE isolation scores after successful pick
            print("\n  Recomputing isolation scores (vial removed from scene)...")
            isolation_scores = recompute_isolation_scores(vial_skip_counts)
            print_top_scores(isolation_scores, count=3, title="After successful placement:")
            
            # Update visualization
            visualize_isolation_scores(vials, vial_skip_counts, label="After grasp")
        else:
            final_pos = get_position(vial)
            final_dist = np.linalg.norm(final_pos[:2] - np.array(target_pos[:2]))
            print(f"  Not in position (dist={final_dist*1000:.1f}mm)")
        
    except Exception as e:
        print(f"  Error: {e}")
        if current_constraint:
            remove_constraint(current_constraint)
            current_constraint = None
        import traceback
        traceback.print_exc()

# Final statistics
print("\n" + "="*70)
print(" SINGULATION COMPLETE")
print("="*70)
print(f"\n RESULTS:")
for vtype in ['small', 'medium', 'large']:
    total = VIAL_SPECS[vtype]['count']
    placed = placed_counts[vtype]
    print(f"   {vtype.upper()}: {placed}/{total} placed ({placed/total*100:.0f}%)")

total_vials = sum(spec['count'] for spec in VIAL_SPECS.values())
total_placed = sum(placed_counts.values())
print(f"\n   OVERALL: {total_placed}/{total_vials} placed "
      f"({total_placed/total_vials*100:.0f}%)")

# Clean up debug text (automatic - debug texts have lifeTime set)
print("\nSimulation complete!")

print("\n Demo complete! Press Ctrl+C to exit")

try:
    while True:
        m.step_simulation()
        time.sleep(0.01)
except KeyboardInterrupt:
    print("\n bye!")

