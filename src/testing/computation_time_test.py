# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
# make sure mujoco and nvidia will be found
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + \
                                ':/workspace/.mujoco/mujoco210/bin:/usr/local/nvidia/lib:/usr/lib/nvidia'
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '/workspace/.mujoco/mujoco210/'
os.environ['MUJOCO_GL'] = 'egl'
# set to glfw if trying to render locally with a monitor
# os.environ['MUJOCO_GL'] = 'glfw'

from mjrl.utils.gym_env import GymEnv
import mujoco_py
import mj_envs
import time

e = GymEnv('hammer-v0')

# when program is init, the first few renders will be a bit slower,
# so we first run 1000 renders without recording their time
for i in range(1000):
    img = e.env.sim.render(width=84, height=84, mode='offscreen', camera_name='top')

# now try to know how much time it takes for 1000 rendering
st = time.time()
for i in range(1000):
    img = e.env.sim.render(width=84, height=84, mode='offscreen', camera_name='top')
et = time.time()

# when mujoco uses gpu to render (when things work correctly, should be fast)
# should take < 0,5 seconds
# if your GPU is not working correct, it might be 10x slower
# if it is slow, then sth is wrong, either the program can't see your gpu
# or your gpu is available, but the old mujoco failed to use it
time_used = et-st
if time_used < 0.5:
    print("time used: %.3f seconds, looks correct!" % time_used)
else:
    print("time used: %.3f seconds." % time_used)
    print("WARNING!!!!! SLOWER THAN EXPECTED, YOUR MUJOCO RENDERING MIGHT NOT WORK CORRECTLY!")

print("Depends on hardware but typically should be < 0.5 seconds.")


