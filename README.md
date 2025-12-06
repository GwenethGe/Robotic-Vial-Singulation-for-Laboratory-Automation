# Demo - Simple Vial Singulation System

This is a **simplified** version of the vial singulation system with:
- **Same modular structure** as `final_project/`
- **Same scene setup** as `final_project/` (box, trays, vial counts)
- **Simple algorithm** from `demo_final_complete copy.py` (constraint-based, no grasp candidates)

## Differences from `final_project/`

### Algorithm Simplicity
- **No grasp candidates**: Uses only simple top-down grasps
- **No Ferrari-Canny scoring**: No wrench space analysis
- **No isolation checking**: No multi-object grasp detection
- **No push strategy**: No singulation push for crowded vials
- **Constraint-based**: Relies on PyBullet constraints for attachment

### File Structure
```
demo/
├── config.py           # Configuration parameters
├── scene_builder.py    # Scene creation (table, box, trays, vials)
├── robot_control.py    # Robot control functions
├── task_logic.py       # Task planning logic
├── main.py            # Main entry point
└── README.md          # This file
```

## Key Features

1. **3 Vial Sizes**: Small (8mm), Medium (10mm), Large (12mm)
2. **3 Matching Trays**: Color-coded with appropriate slot sizes
3. **Simple CV Tracking**: Position updates via `get_position()`
4. **Size-Adaptive Grasping**: Gripper width adjusts to vial size
5. **Constraint Attachment**: PyBullet constraints for secure transport

## Running the Demo

```bash
cd demo
python3 main.py
```

## Algorithm Overview

For each vial:
1. **Detect** vial size and position (simulated CV)
2. **Approach** from above (25cm)
3. **Open gripper** to 3x vial diameter
4. **Descend** in stages (70% → 40% → 10% → 0%)
5. **Close gripper** to 90% of vial radius
6. **Create constraint** to attach vial to gripper
7. **Lift** in 5cm increments (with verification)
8. **Transport** to matching tray slot
9. **Descend** and release
10. **Retreat** to safe height

## Scene Setup (Same as final_project/)

Key parameters in `config.py`:
- **Table**: Surface at z=0.76m
- **Box**: 50cm × 50cm × 12cm, centered at [0.45, 0.0], sits ON table
- **Box bottom**: 2cm thick, on table surface
- **Trays**: 3 trays sitting ON table
  - Small: 6.5cm slots, 3.5cm walls
  - Medium: 7cm slots, 3.5cm walls
  - Large: 4.5cm holes, 3cm thick inner walls, 5.5cm walls
- **Vials**: 4 of each type (12 total), randomly placed in box
- **Physics**: Linear damping 0.8, angular damping 0.8, friction 3.0
- **Gripper**: Force 800N, opens to 3x radius, closes to 85% radius

## Comparison Table

| Feature | `demo/` | `final_project/` |
|---------|---------|------------------|
| Grasp candidates | ❌ No | ✅ 48 candidates |
| Ferrari-Canny scoring | ❌ No | ✅ Yes |
| Isolation checking | ❌ No | ✅ Yes |
| Push strategy | ❌ No | ✅ Yes |
| Home position reset | ❌ No | ✅ Yes |
| Collision filtering | ❌ No | ✅ Yes |
| Grasp stabilization | ❌ No | ✅ Yes (teleport) |
| Structure | ✅ Modular | ✅ Modular |
| Simplicity | ✅ Simple | ❌ Advanced |

