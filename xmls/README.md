# XML Models Directory

This directory contains all MuJoCo XML model files for the Rubik's Cube manipulation project.

## Main Models

### `bidexhands.xml`
- **Primary model file** for the bidexhands Rubik's cube environment
- Includes both shadow hands and the cube
- References: `left_hand.xml`, `right_hand.xml`, `cube_rad.xml`

### `left_hand.xml` & `right_hand.xml`
- **Shadow Hand models** for left and right hands
- Contains detailed hand geometry, joints, and actuators
- Used by the main bidexhands environment

### `cube_rad.xml` & `cube.xml`
- **Rubik's cube models** with different configurations
- `cube_rad.xml`: Cube with radius-based geometry
- `cube.xml`: Standard cube model

## Additional Models

### `hello.xml`
- Simple test model for MuJoCo setup verification

## Usage

When running training or evaluation, specify the XML path relative to the project root:

```bash
# Training
python utils/train_bidexhands_ppo.py --xml_path xmls/bidexhands.xml

# Evaluation
python utils/evaluate_trained_model.py --model_path saved_models/model.pth
```

## File Dependencies

```
bidexhands.xml
├── left_hand.xml
├── right_hand.xml
└── cube_rad.xml
```

All XML files in this directory can reference each other using relative paths since they're in the same folder.
