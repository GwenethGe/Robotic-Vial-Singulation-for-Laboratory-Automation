"""
Configuration parameters for vial singulation system (Simple version)
Same scene setup as final_project/
"""

# Table height
TABLE_HEIGHT = 0.76  # Table surface Z coordinate

# Vial specifications
VIAL_SPECS = {
    'small': {
        'radius': 0.008,   # 8mm
        'height': 0.035,   # 35mm
        'mass': 0.018,     # 18g
        'color': [1.0, 0.3, 0.3, 0.9],  # Red
        'count': 4
    },
    'medium': {
        'radius': 0.010,   # 10mm
        'height': 0.045,   # 45mm
        'mass': 0.025,     # 25g
        'color': [0.3, 1.0, 0.3, 0.9],  # Green
        'count': 4
    },
    'large': {
        'radius': 0.012,   # 12mm
        'height': 0.055,   # 55mm
        'mass': 0.035,     # 35g
        'color': [0.3, 0.3, 1.0, 0.9],  # Blue
        'count': 4
    }
}

# Box configuration (same as final_project/)
# Box sits ON the table
BOX_SIZE = [0.50, 0.50, 0.12]  # 50cm x 50cm x 12cm
BOX_BOTTOM_THICKNESS = 0.020   # 2cm
BOX_POS = [0.45, 0.0, TABLE_HEIGHT + BOX_SIZE[2]/2]  # Center of box

# Tray configurations (same as final_project/)
# Trays sit ON the table
TRAY_WALL_HEIGHT = 0.035  # 3.5cm
LARGE_TRAY_WALL_HEIGHT = 0.055  # 5.5cm - taller for large round-bottom tubes

TRAY_CONFIGS = {
    'small': {
        'position': [-0.35, -0.20, TABLE_HEIGHT],  # Original position
        'rows': 2,
        'cols': 2,
        'slot_size': 0.065,  # 6.5cm (gripper ~56mm needs clearance)
        'wall_height': TRAY_WALL_HEIGHT,
        'color': [1.0, 0.2, 0.2, 1.0]
    },
    'medium': {
        'position': [-0.35, 0.0, TABLE_HEIGHT],  # Original position
        'rows': 2,
        'cols': 2,
        'slot_size': 0.070,  # 7cm (gripper ~60mm needs clearance)
        'wall_height': TRAY_WALL_HEIGHT,
        'color': [0.2, 1.0, 0.2, 1.0]
    },
    'large': {
        'position': [-0.35, 0.20, TABLE_HEIGHT],  # Original position
        'rows': 2,
        'cols': 2,
        'slot_size': 0.045,  # 4.5cm hole size
        'inner_wall_width': 0.030,  # 3cm thick inner walls (gripper=64mm, need spacing)
        'wall_height': LARGE_TRAY_WALL_HEIGHT,  # Taller walls to support round bottom
        'color': [0.2, 0.2, 1.0, 0.9]
    }
}

# Physics parameters
VIAL_FRICTION = 3.0  # Normal friction
LINEAR_DAMPING = 0.8
ANGULAR_DAMPING = 0.8

# Gripper parameters
GRIPPER_OPEN_MULTIPLIER = 3.0  # Open to 3x the vial diameter
GRIPPER_CLOSE_MULTIPLIER = 0.9  # Close to 90% of radius
GRIPPER_FORCE = 100  # Newton (appropriate for 18-35g vials, ~3-10x object weight)
GRIPPER_CENTER_Y_OFFSET = 0.035  # 35mm - gripper center is Y+ from right finger

# Control parameters
PLACEMENT_TOLERANCE = 0.040  # 4cm tolerance for successful placement
MIN_GRIP_MOVEMENT = 0.001  # 1mm minimum movement to confirm grip
MIN_LIFT_HEIGHT = 0.02  # 2cm minimum lift to confirm grasp

# Grasp strategy selection
USE_MULTI_CANDIDATE_GRASPS = True  # True = advanced (multi-candidate), False = simple
NUM_GRASP_ROTATIONS = 8  # Number of rotation angles to try
NUM_GRASP_HEIGHTS = 3    # Number of height offsets to try
NUM_GRASP_APPROACHES = 2 # Number of approach angles to try
# Total candidates = 8 * 3 * 2 = 48 per vial

# Collision detection
ENABLE_COLLISION_CHECK = False  # True = check collisions during descent, False = disable
COLLISION_THRESHOLD = 50  # Number of contact points to trigger collision abort (higher = more tolerant)

# Simulation speed
REAL_TIME_MODE = True  # True = real-time (fast), False = step-by-step (accurate)
TIME_STEP = 1./240.    # 240 Hz for real-time mode

