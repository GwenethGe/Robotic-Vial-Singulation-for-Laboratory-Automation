"""
Scene creation functions (Simple version)
Same scene setup as final_project/
"""
import mengine as m
import numpy as np
import pybullet as p
import os
from scipy.spatial.transform import Rotation
from config import *

# Path to real tube model (optional)
TUBE_URDF_PATH = os.path.join(os.path.dirname(__file__), "assets", "tube.urdf")
USE_REAL_TUBE_MODEL = os.path.exists(TUBE_URDF_PATH)

def create_table():
    """Create table - surface at TABLE_HEIGHT"""
    # Table is a thin box. Surface is at TABLE_HEIGHT.
    # Box center is at TABLE_HEIGHT - half_thickness
    table_thickness = 0.01  # 1cm thick
    return m.Shape(
        m.Box(half_extents=[1.2, 1.2, table_thickness/2]),
        static=True,
        mass=0,
        position=[0, 0, TABLE_HEIGHT - table_thickness/2],
        rgba=[0.85, 0.75, 0.6, 1.0]
    )

def create_box():
    """Create box with solid bottom and walls - sits ON table"""
    pieces = []
    
    # Solid bottom - sits on table surface
    # Bottom center Z = TABLE_HEIGHT + thickness/2
    bottom = m.Shape(
        m.Box(half_extents=[BOX_SIZE[0]/2 + 0.015, 
                           BOX_SIZE[1]/2 + 0.015, 
                           BOX_BOTTOM_THICKNESS/2]),
        static=True,
        mass=0,
        position=[BOX_POS[0], BOX_POS[1], 
                 TABLE_HEIGHT + BOX_BOTTOM_THICKNESS/2],
        rgba=[0.35, 0.20, 0.08, 1.0]
    )
    pieces.append(bottom)
    
    # Four walls - on top of bottom
    wall_thickness = 0.012
    wall_center_z = TABLE_HEIGHT + BOX_BOTTOM_THICKNESS + BOX_SIZE[2]/2
    
    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
        if dx != 0:
            extents = [wall_thickness/2, BOX_SIZE[1]/2, BOX_SIZE[2]/2]
            pos = [BOX_POS[0] + dx*BOX_SIZE[0]/2, BOX_POS[1], wall_center_z]
        else:
            extents = [BOX_SIZE[0]/2, wall_thickness/2, BOX_SIZE[2]/2]
            pos = [BOX_POS[0], BOX_POS[1] + dy*BOX_SIZE[1]/2, wall_center_z]
        
        wall = m.Shape(
            m.Box(half_extents=extents),
            static=True,
            mass=0,
            position=pos,
            rgba=[0.45, 0.28, 0.12, 1.0]
        )
        pieces.append(wall)
    
    return pieces

def create_tray(vtype):
    """Create tray with grid - sits ON table
    
    Layout for 2x2 with inner_wall_width:
    [outer_wall][  slot  ][inner_wall][  slot  ][outer_wall]
    
    Total width = 2*outer_wall + 2*slot + 1*inner_wall
    """
    config = TRAY_CONFIGS[vtype]
    tray_pos = config['position']
    rows = config['rows']
    cols = config['cols']
    slot_size = config['slot_size']
    color = config['color']
    wall_height = config.get('wall_height', TRAY_WALL_HEIGHT)
    
    # Wall widths
    outer_wall_width = 0.012
    inner_wall_width = config.get('inner_wall_width', 0.012)
    
    # Total frame size
    # = outer_wall + (cols * slot) + ((cols-1) * inner_wall) + outer_wall
    frame_width_x = outer_wall_width * 2 + cols * slot_size + (cols - 1) * inner_wall_width
    frame_width_y = outer_wall_width * 2 + rows * slot_size + (rows - 1) * inner_wall_width
    frame_half_x = frame_width_x / 2
    frame_half_y = frame_width_y / 2
    
    pieces = []
    wall_center_z = TABLE_HEIGHT + wall_height / 2
    
    # Outer walls (4 sides)
    # Top and bottom (Y direction)
    for dy in [1, -1]:
        extents = [frame_half_x, outer_wall_width / 2, wall_height / 2]
        pos = [tray_pos[0], 
               tray_pos[1] + dy * (frame_half_y - outer_wall_width / 2), 
               wall_center_z]
        piece = m.Shape(m.Box(half_extents=extents), static=True, mass=0, position=pos, rgba=color)
        pieces.append(piece)
    
    # Left and right (X direction)
    for dx in [1, -1]:
        extents = [outer_wall_width / 2, frame_half_y - outer_wall_width, wall_height / 2]
        pos = [tray_pos[0] + dx * (frame_half_x - outer_wall_width / 2), 
               tray_pos[1], 
               wall_center_z]
        piece = m.Shape(m.Box(half_extents=extents), static=True, mass=0, position=pos, rgba=color)
        pieces.append(piece)
    
    # Inner walls (grid dividers)
    # Horizontal inner walls (between rows)
    for i in range(1, rows):
        # Position: start from bottom, add i*(slot + inner_wall) - inner_wall/2
        y = tray_pos[1] - frame_half_y + outer_wall_width + i * slot_size + (i - 0.5) * inner_wall_width
        piece = m.Shape(
            m.Box(half_extents=[frame_half_x - outer_wall_width, inner_wall_width / 2, wall_height / 2]),
            static=True, mass=0,
            position=[tray_pos[0], y, wall_center_z],
            rgba=color
        )
        pieces.append(piece)
    
    # Vertical inner walls (between columns)
    for i in range(1, cols):
        x = tray_pos[0] - frame_half_x + outer_wall_width + i * slot_size + (i - 0.5) * inner_wall_width
        piece = m.Shape(
            m.Box(half_extents=[inner_wall_width / 2, frame_half_y - outer_wall_width, wall_height / 2]),
            static=True, mass=0,
            position=[x, tray_pos[1], wall_center_z],
            rgba=color
        )
        pieces.append(piece)
    
    return pieces

def create_vials(use_real_model=False, use_test_scene=False, num_vials=30):
    """Create all vials (Full Box Mixed) - inside the box
    
    Args:
        use_real_model: If True, use the real tube.urdf model (for 'large' type)
        use_test_scene: If True, create a predefined test scene for border/angle testing
        num_vials: Total number of vials to create (default 30, distributed evenly across sizes)
    """
    vials = []
    vial_types = []
    
    # Box interior floor = TABLE_HEIGHT + BOX_BOTTOM_THICKNESS
    box_floor_z = TABLE_HEIGHT + BOX_BOTTOM_THICKNESS
    
    # Check if real tube model should be used
    use_real = use_real_model and USE_REAL_TUBE_MODEL
    if use_real:
        print(f"  Using real tube model from: {TUBE_URDF_PATH}")
    
    # Test scene: predefined positions and orientations to test border cases
    if use_test_scene:
        print("  Creating TEST SCENE with border and angle challenges...")
        test_vials = [
            # Format: (vtype, x, y, axis_direction_xyz, tilt_degrees, description)
            
            # Border cases - vials near walls with axis pointing toward wall
            ('medium', 0.25, 0.0, [-1, 0, 0], 60, 'Left wall, axis→left'),
            ('medium', 0.65, 0.0, [1, 0, 0], 60, 'Right wall, axis→right'),
            ('medium', 0.45, -0.20, [0, -1, 0], 60, 'Front wall, axis→front'),
            ('medium', 0.45, 0.20, [0, 1, 0], 60, 'Back wall, axis→back'),
            
            # Corner cases - worst case scenarios
            ('small', 0.25, -0.20, [-1, -1, 0], 45, 'Left-front corner, axis→corner'),
            ('small', 0.65, 0.20, [1, 1, 0], 45, 'Right-back corner, axis→corner'),
            
            # Border with parallel axis (easier cases)
            ('medium', 0.25, 0.10, [0, 1, 0], 45, 'Left wall, axis∥wall'),
            ('medium', 0.65, -0.10, [0, -1, 0], 45, 'Right wall, axis∥wall'),
            
            # Center cases - various tilt angles
            ('small', 0.45, 0.0, [0, 0, 1], 0, 'Center, vertical'),
            ('medium', 0.40, 0.05, [1, 0, 0], 30, 'Center, 30° tilt'),
            ('medium', 0.50, -0.05, [0, 1, 0], 45, 'Center, 45° tilt'),
            ('large', 0.50, 0.10, [1, 1, 0], 60, 'Center, 60° tilt'),
            
            # Moderate border cases (still challenging but not extreme)
            ('small', 0.30, 0.15, [-1, 0.5, 0], 40, 'Near left wall, angled'),
            ('small', 0.60, -0.15, [1, -0.5, 0], 40, 'Near right wall, angled'),
        ]
        
        for vtype, x, y, axis_xy, tilt, desc in test_vials:
            spec = VIAL_SPECS[vtype]
            
            # Position (with small random Z drop)
            z = box_floor_z + spec['height']/2 + np.random.uniform(0.02, 0.05)
            
            # Create orientation from axis and tilt
            # axis_xy defines the XY component of the vial axis
            # tilt is the angle from vertical
            tilt_rad = np.radians(tilt)
            
            # Normalize XY direction
            xy_dir = np.array(axis_xy[:2])
            if np.linalg.norm(xy_dir) > 0:
                xy_dir = xy_dir / np.linalg.norm(xy_dir)
            else:
                xy_dir = np.array([1, 0])
            
            # Vial axis in 3D
            axis = np.array([
                xy_dir[0] * np.sin(tilt_rad),
                xy_dir[1] * np.sin(tilt_rad),
                np.cos(tilt_rad)
            ])
            axis = axis / np.linalg.norm(axis)
            
            # Convert to quaternion
            # Align [0,0,1] to axis
            z_axis = np.array([0, 0, 1])
            if np.abs(np.dot(axis, z_axis)) < 0.9999:
                rot_axis = np.cross(z_axis, axis)
                rot_axis = rot_axis / np.linalg.norm(rot_axis)
                angle = np.arccos(np.dot(z_axis, axis))
                orientation = Rotation.from_rotvec(angle * rot_axis).as_quat()
            else:
                orientation = [0, 0, 0, 1]  # No rotation needed
            
            print(f"    Test vial: {desc}")
            print(f"      Type: {vtype}, Pos: [{x:.2f}, {y:.2f}], Axis: [{axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f}]")
            
            # Create vial (same logic as random scene)
            if use_real and vtype == 'large':
                scale = spec['radius'] * 2 / 0.018
                vial_id = p.loadURDF(
                    TUBE_URDF_PATH,
                    basePosition=[x, y, z],
                    baseOrientation=orientation,
                    globalScaling=scale,
                    useFixedBase=False
                )
                p.changeDynamics(vial_id, -1, 
                               mass=spec['mass'],
                               lateralFriction=VIAL_FRICTION,
                               spinningFriction=1.0,
                               rollingFriction=1.0,
                               linearDamping=LINEAR_DAMPING,
                               angularDamping=ANGULAR_DAMPING,
                               contactStiffness=10000,
                               contactDamping=100)
                vials.append(vial_id)
            else:
                vial = m.Shape(
                    m.Cylinder(radius=spec['radius'], length=spec['height']),
                    static=False,
                    mass=spec['mass'],
                    position=[x, y, z],
                    orientation=orientation,
                    rgba=spec['color']
                )
                vial.set_whole_body_frictions(
                    lateral_friction=VIAL_FRICTION,
                    spinning_friction=1.0,
                    rolling_friction=1.0
                )
                p.changeDynamics(vial.body, -1,
                               linearDamping=LINEAR_DAMPING,
                               angularDamping=ANGULAR_DAMPING)
                vials.append(vial)
            
            vial_types.append(vtype)
        
        print(f"  Created {len(vials)} test vials")
        return vials, vial_types
    
    # ========== RANDOM SCENE (Original) ==========
    # Distribute vials evenly across three sizes
    vials_per_size = num_vials // 3
    remainder = num_vials % 3
    counts = {
        'small': vials_per_size + (1 if remainder > 0 else 0),
        'medium': vials_per_size + (1 if remainder > 1 else 0),
        'large': vials_per_size
    }
    
    # Debug: Print first random number to verify seed is working
    if num_vials > 0:
        test_random = np.random.random()
        print(f"   First random number: {test_random:.6f} (should be different each run)")
    
    for vtype, count in counts.items():
        spec = VIAL_SPECS[vtype]
        for i in range(count):
            margin = spec['radius'] * 4
            x = np.random.uniform(BOX_POS[0] - BOX_SIZE[0]/2 + margin, 
                                 BOX_POS[0] + BOX_SIZE[0]/2 - margin)
            y = np.random.uniform(BOX_POS[1] - BOX_SIZE[1]/2 + margin,
                                 BOX_POS[1] + BOX_SIZE[1]/2 - margin)
            # Vial center Z = floor + half height + LARGER random drop for more variation
            # Increased range from 0.02-0.10 to 0.05-0.20 for more randomness
            z = box_floor_z + spec['height']/2 + np.random.uniform(0.05, 0.20)
            
            # Random orientation with MORE variation
            # Use larger angle range and ensure axis is truly random
            angle = np.random.uniform(0, 2*np.pi)
            # Generate random axis on unit sphere (more uniform than randn)
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.arccos(2 * np.random.random() - 1)
            axis = np.array([np.sin(phi) * np.cos(theta),
                           np.sin(phi) * np.sin(theta),
                           np.cos(phi)])
            orientation = Rotation.from_rotvec(angle * axis).as_quat()
            
            # Use real tube model for 'large' type if available
            if use_real and vtype == 'large':
                # Load URDF model - returns PyBullet body ID
                # Real tube is 18mm diameter, 105mm height
                # Scale to match our 'large' spec (12mm radius = 24mm diameter)
                scale = spec['radius'] * 2 / 0.018  # target_diameter / model_diameter
                
                vial_id = p.loadURDF(
                    TUBE_URDF_PATH,
                    basePosition=[x, y, z],
                    baseOrientation=orientation,
                    globalScaling=scale,
                    useFixedBase=False
                )
                # Set physics properties with collision margin to prevent tunneling
                p.changeDynamics(vial_id, -1, 
                               mass=spec['mass'],
                               lateralFriction=VIAL_FRICTION,
                               spinningFriction=1.0,
                               rollingFriction=1.0,
                               linearDamping=LINEAR_DAMPING,
                               angularDamping=ANGULAR_DAMPING,
                               contactStiffness=10000,
                               contactDamping=100)
                
                # Set collision margin to prevent penetration
                p.setCollisionFilterPair(vial_id, -1, -1, -1, enableCollision=True)
                
                vials.append(vial_id)  # Store PyBullet ID directly
            else:
                # Create simple cylinder vial
                vial = m.Shape(
                    m.Cylinder(radius=spec['radius'], length=spec['height']),
                    static=False,
                    mass=spec['mass'],
                    position=[x, y, z],
                    orientation=orientation,
                    rgba=spec['color']
                )
                
                vial.set_whole_body_frictions(
                    lateral_friction=VIAL_FRICTION, 
                    spinning_friction=1.0, 
                    rolling_friction=1.0
                )
                p.changeDynamics(vial.body, -1, 
                               linearDamping=LINEAR_DAMPING, 
                               angularDamping=ANGULAR_DAMPING)
                
                vials.append(vial)
            
            vial_types.append(vtype)
    
    return vials, vial_types

def create_scene(use_real_tube_model=False, use_test_scene=False, num_vials=30, render=False):
    """Create complete scene
    
    Args:
        use_real_tube_model: If True, use real tube.urdf for 'large' vials
        use_test_scene: If True, create predefined test scene for border/angle testing
        num_vials: Total number of vials to create (default 30)
        render: If True, use GUI mode; if False, use DIRECT mode (no GUI, for batch experiments)
    """
    print("\n[1/5] Creating environment...")
    env = m.Env(render=render)
    
    print("[2/5] Creating table...")
    table = create_table()
    
    print("[3/5] Creating box...")
    box_pieces = create_box()
    
    print("[4/5] Creating trays...")
    trays = {}
    for vtype in ['small', 'medium', 'large']:
        trays[vtype] = create_tray(vtype)
    
    if use_test_scene:
        print("[5/5] Creating TEST SCENE vials...")
    else:
        print(f"[5/5] Creating {num_vials} vials (Full Box Mixed - Random)...")
    vials, vial_types = create_vials(use_real_model=use_real_tube_model, 
                                      use_test_scene=use_test_scene,
                                      num_vials=num_vials)
    
    print(f"   Created {len(vials)} vials")
    if use_real_tube_model and USE_REAL_TUBE_MODEL:
        print(f"   Using REAL tube model for large vials")
    
    # Return box_pieces and table for collision checking
    return env, vials, vial_types, box_pieces, table

