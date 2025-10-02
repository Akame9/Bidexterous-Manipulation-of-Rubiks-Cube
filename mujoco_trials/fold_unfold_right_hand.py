# fold_unfold_shadow_hand.py
# Requires: pip install mujoco
# Optional viewer: pip install glfw

import time
import math
import numpy as np
import mujoco as mj

# Try to import the viewer; fall back to headless if not available
try:
    import mujoco.viewer as mjv
    HAS_VIEWER = True
except Exception:
    print("NO VIEWER : mujoco.viewer not available; running headless")
    HAS_VIEWER = False


XML_PATH = "bidexhands.xml"  # Path to the MuJoCo XML model file


def ease_in_out(t):
    """Smooth step (0→1) with zero velocity at ends."""
    return 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, t)))


def pick_targets(model):
    """
    Compute 'fold' and 'unfold' target control values for each actuator
    based on actuator names and their ctrlrange.
    """
    fold_targets = np.zeros(model.nu)
    rest_targets = np.zeros(model.nu)

    # Convenience: lookup ranges and names once
    ctrl_ranges = model.actuator_ctrlrange.copy()  # shape [nu, 2]
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]

    # Heuristics:
    # - For main finger flexors (proximal + middle_distal tendon), drive to upper bound for "fold".
    # - For thumb flexors (THJ4, THJ1), drive to upper bound.
    # - Knuckle ab/adduction we keep near neutral (0).
    # - Metacarpal (LFJ5) small flex toward upper bound for fold.
    # - Wrist stays neutral.
    # Everything "unfold" goes to the lower bound if range starts at 0, else neutral (0) if symmetric.

    def neutral_from_range(lo, hi):
        # If symmetric around 0, pick 0; otherwise pick midpoint.
        if abs(hi + lo) < 1e-6:
            return 0.0
        return 0.5 * (lo + hi)

    for i, name in enumerate(names):
        lo, hi = ctrl_ranges[i]

        # Default: rest target tries neutral; fold tries midpoint→upper
        rest = neutral_from_range(lo, hi)
        fold = rest #hi  # optimistic fold is the upper bound (most flexion in most of your defs)

        # Fine-tune by actuator name
        # Wrist
        if name in {"rh_A_WRJ1", "rh_A_WRJ2"}:
            rest = neutral_from_range(lo, hi)
            fold = rest  # keep wrist neutral

        # Knuckles (ab/adduction) — keep neutral
        elif any(k in name for k in ["rh_A_FFJ4", "rh_A_MFJ4", "rh_A_RFJ4", "rh_A_LFJ4"]):
            rest = neutral_from_range(lo, hi)
            fold = rest

        # Metacarpal (little finger base) — small flex
        elif name == "rh_A_LFJ5":
            rest = neutral_from_range(lo, hi)
            fold = lo + 0.8 * (hi - lo)

        # Proximal flexors — fold hard
        elif any(k in name for k in ["rh_A_FFJ3", "rh_A_MFJ3", "rh_A_RFJ3", "rh_A_LFJ3"]):
            rest = lo  # these have ranges like [-0.26, 1.57]; open ≈ lower end
            fold = hi  # closed ≈ upper end

        # Tendon controls for middle+distal flexion — fold hard
        elif name.endswith("J0") and name.startswith("rh_A_"):  # rh_A_*J0
            rest = lo  # usually [0, 1.57] → open at 0
            fold = hi

        # Thumb:
        # THJ4 (proximal), THJ1 (distal) → fold hard to upper bound
        elif name in {"rh_A_THJ4", "rh_A_THJ1"}:
            rest = lo if lo <= 0.0 <= hi else neutral_from_range(lo, hi)
            fold = hi

        # THJ5 (base roll), THJ3 (hub), THJ2 (middle yaw) → modest curl
        elif name in {"rh_A_THJ5", "rh_A_THJ3", "rh_A_THJ2"}:
            rest = neutral_from_range(lo, hi)
            # go 70% toward whichever side has larger magnitude positive hi
            fold = lo + 0.7 * (hi - lo)

        # Everything else: keep the default choice above

        rest_targets[i] = rest
        fold_targets[i] = fold

    return rest_targets, fold_targets, names

def pick_targets_1(model):
    """
    Compute 'fold' and 'unfold' target control values for each actuator
    based on actuator names and their ctrlrange.
    """
    fold_targets = np.zeros(model.nu)
    rest_targets = np.zeros(model.nu)

    # Convenience: lookup ranges and names once
    ctrl_ranges = model.actuator_ctrlrange.copy()  # shape [nu, 2]
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]

    def neutral_from_range(lo, hi):
        # If symmetric around 0, pick 0; otherwise pick midpoint.
        if abs(hi + lo) < 1e-6:
            return 0.0
        return 0.5 * (lo + hi)

    for i, name in enumerate(names):
        lo, hi = ctrl_ranges[i]
        
        rest = neutral_from_range(lo, hi)
        fold = rest 
        
        if name in {"rh_A_WRJ1", "rh_A_WRJ2"}:
            rest = neutral_from_range(lo, hi)
            fold = rest  # keep wrist neutral

        # Knuckles (ab/adduction) — keep neutral
        elif any(k in name for k in ["rh_A_FFJ4", "rh_A_MFJ4", "rh_A_RFJ4", "rh_A_LFJ4"]):
            rest = neutral_from_range(lo, hi)
            fold = rest

        # Metacarpal (little finger base) — small flex
        elif name == "rh_A_LFJ5":
            rest = neutral_from_range(lo, hi)
            fold = lo + 0.8 * (hi - lo)

        # Proximal flexors — fold hard
        elif any(k in name for k in ["rh_A_FFJ3", "rh_A_MFJ3", "rh_A_RFJ3", "rh_A_LFJ3"]):
            rest = lo  # these have ranges like [-0.26, 1.57]; open ≈ lower end
            fold = hi  # closed ≈ upper end

        # Tendon controls for middle+distal flexion — fold hard
        elif name.endswith("J0") and name.startswith("rh_A_"):  # rh_A_*J0
            rest = lo  # usually [0, 1.57] → open at 0
            fold = hi

        # Thumb:
        # THJ4 (proximal), THJ1 (distal) → fold hard to upper bound
        elif name in {"rh_A_THJ4", "rh_A_THJ1"}:
            rest = lo if lo <= 0.0 <= hi else neutral_from_range(lo, hi)
            fold = hi

        # THJ5 (base roll), THJ3 (hub), THJ2 (middle yaw) → modest curl
        elif name in {"rh_A_THJ5", "rh_A_THJ3", "rh_A_THJ2"}:
            rest = neutral_from_range(lo, hi)
            # go 70% toward whichever side has larger magnitude positive hi
            fold = lo + 0.7 * (hi - lo)  

        rest_targets[i] = rest
        fold_targets[i] = fold

    return rest_targets, fold_targets, names

def set_ctrl(data, target_ctrl):
    """Write the entire control vector."""
    data.ctrl[:] = target_ctrl


def interpolate_ctrl(a, b, alpha):
    """Elementwise interpolate (0..1)."""
    return (1 - alpha) * a + alpha * b


def run_headless(model, data, rest, fold, period_s=3.0, settle_s=0.5, steps=3):
    """
    Headless run:
      - Ease from rest → fold over period_s
      - Hold settle_s
      - Ease from fold → rest over period_s
      - Repeat 'steps' times
    """
    dt = model.opt.timestep

    def ramp(start_ctrl, end_ctrl, duration):
        t = 0.0
        while t < duration:
            alpha = ease_in_out(t / duration)
            set_ctrl(data, interpolate_ctrl(start_ctrl, end_ctrl, alpha))
            mj.mj_step(model, data)
            t += dt

    # Initial rest pose
    set_ctrl(data, rest)
    for _ in range(int(0.5 / dt)):
        mj.mj_step(model, data)

    for _ in range(steps):
        # Fold
        ramp(rest, fold, period_s)
        # Settle
        for _ in range(int(settle_s / dt)):
            set_ctrl(data, fold)
            mj.mj_step(model, data)
        # Unfold
        ramp(fold, rest, period_s)
        # Settle
        for _ in range(int(settle_s / dt)):
            set_ctrl(data, rest)
            mj.mj_step(model, data)



def run_with_viewer(model, data, rest, fold, cycle_s=6.0):
    """
    Interactive viewer loop:
      Continuously oscillate between rest and fold with a cosine profile.
      Press ESC to quit.
    """
    with mjv.launch_passive(model, data) as viewer:
        t0 = time.time()
        while viewer.is_running():
            # Smooth cosine oscillation between rest and fold
            phase = (time.time() - t0) * (2 * math.pi / cycle_s)
            alpha = 0.5 * (1 - math.cos(phase))
            set_ctrl(data, interpolate_ctrl(rest, fold, alpha))
            
            # Step physics
            mj.mj_step(model, data)
            
            # Update viewer
            viewer.sync()
            
def set_folded_pose(model, data, fold):
    """
    Immediately set the Shadow Hand to the folded pose.
    No interpolation, just apply 'fold' controls and settle for a few steps.
    """
    # Apply folded control values
    set_ctrl(data, fold)  # slightly less than full fold to avoid singularities
    
    # Step the simulation forward a bit so joints settle
    for _ in range(200):  # ~50 steps is usually enough
        mj.mj_step(model, data)

def pick_targets_from_joint_ranges(model):
    """
    Compute rest and fold targets based on the joint ranges each actuator drives.
    - Joint-based actuators → fold to joint max
    - Tendon-based actuators → fold to actuator ctrl hi
    - Wrist & knuckle actuators → kept neutral
    """
    fold_targets = np.zeros(model.nu)
    rest_targets = np.zeros(model.nu)

    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    ctrl_ranges = model.actuator_ctrlrange.copy()

    def neutral_from_range(lo, hi):
        if abs(lo + hi) < 1e-6:
            return 0.0
        return 0.5 * (lo + hi)

    for i, name in enumerate(names):
        lo, hi = ctrl_ranges[i]
        rest = neutral_from_range(lo, hi)
        fold = hi  # default fallback

        trnid = model.actuator_trnid[i]  # [joint_id, -1] or [tendon_id, -1]
        joint_id = trnid[0]

        if joint_id != -1:  # actuator controls a joint
            jlo, jhi = model.jnt_range[joint_id]
            fold = jhi
            rest = neutral_from_range(jlo, jhi)

        # Special cases
        if name in {"robot0:A_WRJ1", "robot0:A_WRJ0"}:  # wrist
            fold = rest
        elif any(k in name for k in ["rh_A_FFJ4", "rh_A_MFJ4", "rh_A_RFJ4", "rh_A_LFJ4"]):  # knuckles
            fold = rest

        rest_targets[i] = rest
        fold_targets[i] = fold

    return rest_targets, fold_targets, names

def main():
    model = mj.MjModel.from_xml_path(XML_PATH)
    data = mj.MjData(model)

    rest, fold, names = pick_targets(model)

    print("Discovered actuators and fold targets:")
    for i, nm in enumerate(names):
        lo, hi = model.actuator_ctrlrange[i]
        print(f"  {nm:>14s}  range=({lo:+.3f}, {hi:+.3f})  rest={rest[i]:+.3f}  fold={fold[i]:+.3f}")

    if HAS_VIEWER:
        run_with_viewer(model, data, rest, fold, cycle_s=6.0)
        # set_folded_pose(model, data, fold)
        with mjv.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mj.mj_step(model, data)
                viewer.sync()
    else:
        print("No viewer available; running headless cycles...")
        run_headless(model, data, rest, fold, period_s=2.5, settle_s=0.5, steps=4)


if __name__ == "__main__":
    main()
