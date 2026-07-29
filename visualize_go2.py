"""Quick Go2 model visualizer using MuJoCo passive viewer."""
import mujoco
import mujoco.viewer
from pathlib import Path

GO2_SCENE = Path(__file__).resolve().parent / "assets" / "unitree_go2" / "scene.xml"

model = mujoco.MjModel.from_xml_path(str(GO2_SCENE))
data = mujoco.MjData(model)

# Reset to default pose
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

print(f"Go2 model: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators, {model.njnt} joints")
print("Launching viewer... (close window to exit)")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data, nstep=10)
        viewer.sync()
