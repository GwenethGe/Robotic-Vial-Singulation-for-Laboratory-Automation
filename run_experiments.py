#!/usr/bin/env python3
"""
Run multiple trials and collect quantitative data for paper
"""

import sys
import time
import signal
import numpy as np
import pybullet as p
import mengine as m

# Global flag for graceful shutdown
interrupted = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global interrupted
    print("\n\n" + "="*80)
    print("INTERRUPT SIGNAL RECEIVED (Ctrl+C)")
    print("="*80)
    print("Stopping experiment gracefully...")
    print("Current trial will be marked as incomplete.")
    print("="*80 + "\n")
    interrupted = True

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# Import only what we need, not the entire main.py execution
from config import *
from scene_builder import create_scene, create_tray
from robot_control import (
    get_position, get_orientation, get_vial_axis,
    open_gripper, close_gripper_gently,
    move_to_pose, check_robot_collisions,
    create_vial_constraint, remove_constraint,
    singulation_push, calculate_push_trajectory, calculate_vertical_placement_pose,
    shake_box_with_gripper
)
from task_logic import (
    compute_target_slot_position,
    check_placement_success,
    is_tray_full,
    check_grasp_isolation,
    compute_isolation_score
)

# Import multi-candidate system if enabled
if USE_MULTI_CANDIDATE_GRASPS:
    from grasp_candidates import generate_grasp_candidates, select_best_grasp

from experiment_logger import reset_logger

# Constants
MAX_RETRIES = 3  # Maximum retry attempts before skipping

def get_vial_tilt_angle(vial):
    """Calculate vial tilt angle from vertical"""
    # Handle both Body objects and integer IDs
    vial_body_id = vial if isinstance(vial, int) else vial.body
    _, vial_orient = p.getBasePositionAndOrientation(vial_body_id)
    vial_axis = get_vial_axis(vial_orient)
    vertical = np.array([0, 0, 1])
    tilt_rad = np.arccos(np.clip(np.abs(np.dot(vial_axis, vertical)), -1, 1))
    return np.degrees(tilt_rad)

def run_single_trial_with_logging(trial_num, logger, num_vials=30):
    """Run a single trial with logging"""
    
    print(f"\n{'='*80}")
    print(f"STARTING TRIAL {trial_num} - {num_vials} vials, 12 tray slots")
    print(f"{'='*80}\n")
    
    # Set random seed for reproducibility and variation across trials
    # Use trial number + timestamp to ensure different scenes each trial
    seed = int(time.time() * 1000) % (2**31) + trial_num * 1000
    np.random.seed(seed)
    print(f"Random seed: {seed}")
    
    # Start trial logging
    logger.start_trial(trial_num, num_vials)
    
    # Create scene (use DIRECT mode for batch experiments - no GUI)
    print("Creating scene...")
    env, vials, vial_types, box_pieces, table = create_scene(num_vials=num_vials, render=False)
    
    # Create robot
    robot = m.Robot.Panda(position=[0, 0, 0.76])
    robot.motor_gains = 0.03  # Moderate speed
    home_joints = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]  # Safe home configuration
    
    # Create trays
    from scene_builder import create_tray
    trays = {
        'SMALL': create_tray('small'),
        'MEDIUM': create_tray('medium'),
        'LARGE': create_tray('large')
    }
    
    # Map vials to their specs and types
    vial_specs = {}
    vial_type_map = {}
    for i, vial in enumerate(vials):
        vtype = vial_types[i]
        # vtype is already lowercase from scene_builder ('small', 'medium', 'large')
        vial_specs[vial] = VIAL_SPECS[vtype]
        vial_type_map[vial] = vtype.upper()  # Store uppercase for tray access
    
    # Wait for settling
    print("Waiting for vials to settle...")
    for _ in range(500):
        m.step_simulation()
    
    # Initialize tracking
    placed_vials = set()
    skipped_vials = set()
    vial_retry_counts = {v: 0 for v in vials}
    
    # Tray management
    tray_slots = {
        'SMALL': {'tray': trays['SMALL'], 'next_slot': (0, 0), 'placed': []},
        'MEDIUM': {'tray': trays['MEDIUM'], 'next_slot': (0, 0), 'placed': []},
        'LARGE': {'tray': trays['LARGE'], 'next_slot': (0, 0), 'placed': []}
    }
    
    # Failure tracking
    consecutive_failures = 0
    AGGRESSIVE_MODE_THRESHOLD = 5
    SHAKE_BOX_THRESHOLD = 8
    last_shake_failure_count = -1
    
    # Main picking loop
    attempt_count = 0
    max_attempts = len(vials) * 3
    
    while attempt_count < max_attempts:
        attempt_count += 1
        
        # Check for interrupt
        if interrupted:
            print("\n\nExperiment interrupted by user during trial.")
            break
        
        # Check if all trays full (12 slots total)
        total_placed = sum(len(tray_slots[size]['placed']) for size in ['SMALL', 'MEDIUM', 'LARGE'])
        if total_placed >= 12:
            print("\nAll 12 tray slots filled! Task complete.")
            break
        
        # Check if any vials remain
        remaining_vials = [v for v in vials if v not in placed_vials and v not in skipped_vials]
        if not remaining_vials:
            print("\nNo more vials available in box.")
            break
        
        # Compute isolation scores
        isolation_scores = []
        for vial in remaining_vials:
            spec = vial_specs[vial]
            vial_pos = get_position(vial)
            
            # Skip if outside box
            if not (BOX_POS[0] - BOX_SIZE[0]/2 < vial_pos[0] < BOX_POS[0] + BOX_SIZE[0]/2 and
                   BOX_POS[1] - BOX_SIZE[1]/2 < vial_pos[1] < BOX_POS[1] + BOX_SIZE[1]/2):
                continue
            
            # compute_isolation_score expects (vial_index, all_vials_list)
            vial_idx = vials.index(vial)
            score = compute_isolation_score(vial_idx, vials)
            isolation_scores.append((vial, score, spec))
        
        if not isolation_scores:
            print("\nWARNING: No valid vials remaining in box")
            break
        
        # Select next vial
        isolation_scores.sort(key=lambda x: x[1], reverse=True)
        vial, isolation_score, spec = isolation_scores[0]
        
        # Get vial info
        vial_pos = get_position(vial)
        vial_orient = get_orientation(vial)
        tilt_angle = get_vial_tilt_angle(vial)
        
        # Start vial logging
        logger.start_vial(
            vial_id=vials.index(vial),
            size=vial_type_map[vial],
            position=vial_pos,
            orientation_deg=tilt_angle
        )
        
        print(f"\n{'='*60}")
        print(f"Target: Vial {vials.index(vial)} ({vial_type_map[vial]}, isolation={isolation_score:.0f})")
        print(f"Position: [{vial_pos[0]:.3f}, {vial_pos[1]:.3f}, {vial_pos[2]:.3f}]")
        print(f"Tilt: {tilt_angle:.1f}°")
        print(f"Progress: {len(placed_vials)}/12 slots filled, {len(remaining_vials)} vials remaining")
        print(f"{'='*60}")
        
        # Aggressive mode
        MIN_ISOLATION_DISTANCE = 0.04
        if consecutive_failures >= AGGRESSIVE_MODE_THRESHOLD:
            MIN_ISOLATION_DISTANCE = 0.03
            print(f"AGGRESSIVE MODE: Reduced isolation threshold to {MIN_ISOLATION_DISTANCE*100:.0f}cm")
        
        # Box shake mode
        if consecutive_failures >= SHAKE_BOX_THRESHOLD and last_shake_failure_count != consecutive_failures:
            print(f"\nSHAKE MODE: {consecutive_failures} consecutive failures")
            print("Sweeping vials with gripper...")
            shake_box_with_gripper(robot, box_pieces, num_shakes=3)
            logger.log_sweep()
            last_shake_failure_count = consecutive_failures
            for _ in range(200):
                m.step_simulation()
            continue
        
        # Check if too crowded
        vial_idx = vials.index(vial)
        is_isolated, nearest_dist, nearest_idx, crowded_count = check_grasp_isolation(
            vial_pos, vials, vial_idx, min_distance=MIN_ISOLATION_DISTANCE
        )
        
        if not is_isolated:
            print(f"WARNING: Vial too crowded (min distance: {MIN_ISOLATION_DISTANCE*100:.0f}cm, {crowded_count} within 4cm)")
            print("Attempting singulation push...")
            
            # Get nearest vial position
            nearest_vial_pos = get_position(vials[nearest_idx])
            
            # Calculate push trajectory
            push_result = calculate_push_trajectory(vial_pos, nearest_vial_pos, push_distance=0.08)
            
            if push_result:
                push_start, push_end, push_orient = push_result
                push_success = singulation_push(robot, push_start, push_end, push_orient, steps=500)
                logger.log_push()
                
                if not push_success:
                    print("Push failed or vial still crowded")
                
                for _ in range(200):
                    m.step_simulation()
            else:
                print("Could not calculate valid push trajectory")
            
            # Try grasping anyway after push
            is_isolated, _, _, _ = check_grasp_isolation(
                vial_pos, vials, vial_idx, min_distance=MIN_ISOLATION_DISTANCE * 0.7
            )
        
        # Generate grasp candidates
        t_start = time.time()
        candidates = generate_grasp_candidates(
            vial_pos, vial_orient,
            spec['radius'], spec['height'],
            num_rotations=NUM_GRASP_ROTATIONS,
            num_heights=NUM_GRASP_HEIGHTS,
            num_approaches=NUM_GRASP_APPROACHES,
            vial_idx=vial_idx
        )
        t_gen = time.time() - t_start
        logger.log_timing('candidate_generation', t_gen)
        
        # Score candidates
        t_start = time.time()
        best_grasp = select_best_grasp(
            candidates, robot, vial,
            spec['mass'], spec['radius'],
            mu=VIAL_FRICTION,
            top_k=5, verbose=True,
            all_vials=vials, vial_idx=vial_idx
        )
        t_score = time.time() - t_start
        logger.log_timing('candidate_scoring', t_score)
        
        num_valid = sum(1 for c in candidates if c.is_valid)
        logger.log_grasp_candidates(len(candidates), num_valid, 
                                   best_grasp.total_score if best_grasp else 0.0)
        
        if not best_grasp:
            print("ERROR: No valid grasp candidates found!")
            logger.end_vial(success=False, failure_reason="no_valid_candidates")
            consecutive_failures += 1
            
            vial_retry_counts[vial] += 1
            if vial_retry_counts[vial] >= MAX_RETRIES:
                print(f"SKIP: Skipping vial after {MAX_RETRIES} attempts")
                skipped_vials.add(vial)
                robot.control(home_joints)
                for _ in range(100):
                    m.step_simulation()
            continue
        
        print(f"OK: Best grasp score: {best_grasp.total_score:.3f}")
        
        # Execute grasp
        t_start = time.time()
        grasp_pos = best_grasp.position
        gripper_orient = best_grasp.orientation
        
        # Approach
        approach_height = grasp_pos[2] + 0.15
        approach_pos = [grasp_pos[0], grasp_pos[1], approach_height]
        
        print("Approaching...")
        success = move_to_pose(robot, approach_pos, gripper_orient,
                              check_collisions=ENABLE_COLLISION_CHECK,
                              box_pieces=box_pieces, table=table, held_vial=None)
        
        if not success:
            print("ERROR: IK/Approach failed")
            logger.end_vial(success=False, failure_reason="ik_approach_failed")
            consecutive_failures += 1
            vial_retry_counts[vial] += 1
            if vial_retry_counts[vial] >= MAX_RETRIES:
                skipped_vials.add(vial)
                robot.control(home_joints)
            continue
        
        # Open gripper
        is_horizontal = tilt_angle > 45
        open_gripper(robot, vial_radius=spec['radius'], is_horizontal=is_horizontal)
        for _ in range(50):
            m.step_simulation()
        
        # Descend
        print("Descending to grasp...")
        descent_steps = 15
        for i in range(descent_steps + 1):
            t = i / descent_steps
            current_z = approach_height + t * (grasp_pos[2] - approach_height)
            current_pos = [grasp_pos[0], grasp_pos[1], current_z]
            
            move_to_pose(robot, current_pos, gripper_orient,
                        check_collisions=False, box_pieces=box_pieces,
                        table=table, held_vial=None)
            
            for _ in range(10):
                m.step_simulation()
        
        # Close gripper
        if is_horizontal:
            close_amount = spec['radius'] * 2.0
            grip_force = GRIPPER_FORCE * 1.5
        else:
            close_amount = spec['radius'] * 1.8
            grip_force = GRIPPER_FORCE
        
        initial_vial_pos = get_position(vial)
        robot.set_gripper_position([close_amount, close_amount], 
                                   set_instantly=False,
                                   force=grip_force)
        
        for _ in range(100):
            m.step_simulation()
        
        # Check grip
        current_vial_pos = get_position(vial)
        moved_initial = np.linalg.norm(np.array(current_vial_pos) - np.array(initial_vial_pos))
        
        gripper_joints = robot.get_joint_angles(joints=robot.gripper_joints)
        gripper_closed_properly = gripper_joints[0] < (close_amount + 0.005)
        
        if moved_initial < MIN_GRIP_MOVEMENT or not gripper_closed_properly:
            print(f"ERROR: Vial not gripped (moved: {moved_initial*1000:.1f}mm, gripper: {gripper_joints[0]*1000:.1f}mm)")
            logger.end_vial(success=False, failure_reason="vial_not_gripped")
            consecutive_failures += 1
            vial_retry_counts[vial] += 1
            logger.log_retry()
            
            open_gripper(robot, vial_radius=spec['radius'])
            robot.control(home_joints)
            for _ in range(100):
                m.step_simulation()
            
            if vial_retry_counts[vial] >= MAX_RETRIES:
                skipped_vials.add(vial)
            continue
        
        print(f"OK: Vial gripped! (moved: {moved_initial*1000:.1f}mm)")
        
        # Create constraint
        constraint_id = create_vial_constraint(robot, vial)
        if constraint_id is None:
            print("ERROR: Failed to create constraint")
            logger.end_vial(success=False, failure_reason="constraint_failed")
            consecutive_failures += 1
            continue
        
        t_grasp = time.time() - t_start
        logger.log_timing('grasp_execution', t_grasp)
        
        # Transport and place
        t_start = time.time()
        
        # Lift and re-orient
        print("Lifting and re-orienting to vertical...")
        lift_height = 0.20
        vertical_vial_orient = m.get_quaternion([np.pi, 0, 0])
        
        for step in range(20):
            t = (step + 1) / 20
            current_z = grasp_pos[2] + t * lift_height
            lift_pos = [grasp_pos[0], grasp_pos[1], current_z]
            
            move_to_pose(robot, lift_pos, gripper_orient,
                        check_collisions=False, box_pieces=box_pieces,
                        table=table, held_vial=vial)
            
            for _ in range(5):
                m.step_simulation()
            
            # Force vertical if tilted
            check_vial_tilt = get_vial_tilt_angle(vial)
            if check_vial_tilt > 15:
                vial_body_id = vial if isinstance(vial, int) else vial.body
                p.resetBasePositionAndOrientation(
                    vial_body_id, get_position(vial), vertical_vial_orient
                )
                for _ in range(10):
                    m.step_simulation()
        
        # Move to tray
        target_size = vial_type_map[vial]  # This is uppercase like 'SMALL'
        target_tray = tray_slots[target_size]['tray']
        target_slot = tray_slots[target_size]['next_slot']
        row, col = target_slot
        
        # compute_target_slot_position expects lowercase
        target_pos = compute_target_slot_position(target_size.lower(), row, col)
        
        # Move above tray
        above_target = [target_pos[0], target_pos[1], target_pos[2] + 0.20]
        final_orient = m.get_quaternion([np.pi, 0, 0])
        
        success = move_to_pose(robot, above_target, final_orient,
                              check_collisions=False, box_pieces=box_pieces,
                              table=table, held_vial=vial)
        
        if not success:
            print("ERROR: Failed to reach tray")
            remove_constraint(constraint_id)
            logger.end_vial(success=False, failure_reason="tray_unreachable")
            consecutive_failures += 1
            continue
        
        # Descend into slot
        for step in range(15):
            t = (step + 1) / 15
            current_z = above_target[2] + t * (target_pos[2] - above_target[2])
            descent_pos = [target_pos[0], target_pos[1], current_z]
            
            move_to_pose(robot, descent_pos, final_orient,
                        check_collisions=False, box_pieces=box_pieces,
                        table=table, held_vial=vial)
            
            for _ in range(5):
                m.step_simulation()
            
            # Force vertical
            check_vial_tilt = get_vial_tilt_angle(vial)
            if check_vial_tilt > 15:
                vial_body_id = vial if isinstance(vial, int) else vial.body
                p.resetBasePositionAndOrientation(
                    vial_body_id, get_position(vial), vertical_vial_orient
                )
                for _ in range(10):
                    m.step_simulation()
        
        # Release
        remove_constraint(constraint_id)
        open_gripper(robot, vial_radius=spec['radius'])
        
        for _ in range(100):
            m.step_simulation()
        
        # Check placement
        placement_ok = check_placement_success(vial, target_pos)
        
        t_transport = time.time() - t_start
        logger.log_timing('transport', t_transport)
        
        if placement_ok:
            print(f"SUCCESS: Vial placed successfully in {target_size} tray!")
            placed_vials.add(vial)
            tray_slots[target_size]['placed'].append(vial)
            tray_slots[target_size]['next_slot'] = (
                (target_slot[0] + 1) % 2,
                target_slot[1] + (1 if target_slot[0] == 1 else 0)
            )
            consecutive_failures = 0
            vial_retry_counts[vial] = 0
            logger.end_vial(success=True)
        else:
            print("ERROR: Placement failed")
            logger.end_vial(success=False, failure_reason="placement_failed")
            consecutive_failures += 1
        
        # Return home
        robot.control(home_joints)
        for _ in range(100):
            m.step_simulation()
    
    # End trial
    logger.end_trial()
    
    # Cleanup
    p.disconnect()
    
    return logger

def run_experiments(num_trials=10, num_vials=30, experiment_name=None):
    """Run multiple trials and collect data"""
    
    if experiment_name is None:
        experiment_name = f"vial_singulation_{num_trials}trials_{num_vials}vials"
    
    # Initialize logger
    logger = reset_logger(experiment_name)
    
    print(f"\n{'#'*80}")
    print(f"# STARTING EXPERIMENT: {experiment_name}")
    print(f"# Number of trials: {num_trials}")
    print(f"# Vials per trial: {num_vials}")
    print(f"# Mode: DIRECT (no GUI, for batch experiments)")
    print(f"{'#'*80}\n")
    
    # Run trials
    for trial_num in range(1, num_trials + 1):
        # Check for interrupt
        if interrupted:
            print(f"\nExperiment interrupted by user. Completed {trial_num-1}/{num_trials} trials.")
            break
        
        try:
            run_single_trial_with_logging(trial_num, logger, num_vials)
        except KeyboardInterrupt:
            print(f"\n\nKeyboardInterrupt caught in trial {trial_num}. Stopping experiment...")
            break
        except Exception as e:
            print(f"\nERROR: ERROR in trial {trial_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Short break between trials
        time.sleep(2)
    
    # Save results
    print(f"\n{'#'*80}")
    print("# EXPERIMENT COMPLETE")
    print(f"{'#'*80}\n")
    
    logger.print_summary()
    json_file, report_file, csv_file = logger.save_results()
    
    print(f"\nSUCCESS: All results saved!")
    print(f"\nNext steps:")
    print(f"1. Review the report: {report_file}")
    print(f"2. Analyze the CSV: {csv_file}")
    print(f"3. Update paper.tex with real numbers")
    
    return logger

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run vial singulation experiments')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials to run')
    parser.add_argument('--vials', type=int, default=30, help='Vials per trial (30 vials, 12 tray slots)')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    
    args = parser.parse_args()
    
    logger = run_experiments(
        num_trials=args.trials,
        num_vials=args.vials,
        experiment_name=args.name
    )

