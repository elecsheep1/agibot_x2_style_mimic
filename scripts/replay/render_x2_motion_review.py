"""Render offline kinematic motion sheets/videos, without stepping physics."""

# Support direct execution: python scripts/<category>/<tool>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from checks.check_x2_retarget import load_input


def render(path, output, times, video=False):
    model, q, t, *_ = load_input(path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=360, width=480)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135, -15, 2.2
    options = mujoco.MjvOption()
    # Group 1 is collision in some bundles; render authored visual defaults.
    def frame(f):
        data.qpos[:] = q[f]
        mujoco.mj_forward(model, data)
        camera.lookat[:] = q[f, :3] + [0, 0, .05]
        renderer.update_scene(data, camera=camera, scene_option=options)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        # Add a fixed world floor reference even when the XML has no floor geom.
        index = renderer.scene.ngeom
        if not np.any(model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE):
            mujoco.mjv_initGeom(renderer.scene.geoms[index], mujoco.mjtGeom.mjGEOM_PLANE,
                            np.array([2., 2., .001]), np.array([q[f,0], q[f,1], 0.]),
                            np.eye(3).reshape(-1), np.array([.32,.35,.38,1.], dtype=np.float32))
            renderer.scene.ngeom += 1
        im = Image.fromarray(renderer.render().copy())
        draw = ImageDraw.Draw(im)
        draw.rectangle((0,0,480,24), fill='black')
        draw.text((8,5), f'{path.stem}   t={t[f]:.2f}s', fill='white')
        return im
    output.parent.mkdir(parents=True, exist_ok=True)
    if video:
        import imageio.v2 as imageio
        with imageio.get_writer(output, fps=25, codec='libx264', quality=7) as writer:
            for f in range(0,len(q),max(1,round(1/(25*(t[1]-t[0]))))):
                writer.append_data(np.asarray(frame(f)))
    else:
        images = [frame(int(np.argmin(abs(t-time)))) for time in times]
        sheet = Image.new('RGB', (480*3, 360*((len(images)+2)//3)), 'white')
        for i, im in enumerate(images):sheet.paste(im, ((i%3)*480,(i//3)*360))
        sheet.save(output)
    renderer.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--times',type=float,nargs='+',default=[0,1,2,3,4,5])
    p.add_argument('--video',action='store_true')
    a=p.parse_args();render(a.motion,a.output,a.times,a.video)
