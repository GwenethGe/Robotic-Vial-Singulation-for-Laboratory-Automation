# Robotic Vial Singulation System for Laboratory Automation



## Key Features

1. **3 Vial Sizes**: Small (8mm), Medium (10mm), Large (12mm)
2. **3 Matching Trays**: Color-coded with appropriate slot sizes
3. **Simple CV Tracking**: Position updates via `get_position()`
4. **Size-Adaptive Grasping**: Gripper width adjusts to vial size
5. **Constraint Attachment**: PyBullet constraints for secure transport

## Try this out

```bash
python3 main.py
```


## Scene Setup

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


