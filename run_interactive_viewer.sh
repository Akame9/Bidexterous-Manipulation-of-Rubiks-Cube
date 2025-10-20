#!/bin/bash
# Quick launch script for the interactive reward viewer

echo "=========================================="
echo "  Interactive Reward Viewer Launcher"
echo "=========================================="
echo ""
echo "This script will launch the interactive viewer"
echo "where you can control hand actuators and see"
echo "real-time contact forces and rewards."
echo ""
echo "Controls:"
echo "  Arrow keys: Adjust actuator values"
echo "  TAB: Select next actuator"
echo "  R: Reset current actuator"
echo "  SPACE: Reset all to neutral"
echo "  H: Toggle help"
echo "  ESC: Exit"
echo ""
echo "Starting viewer..."
echo ""

# Run the GUI version (recommended)
python3 interactive_reward_viewer_gui.py "$@"

