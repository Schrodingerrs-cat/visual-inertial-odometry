"""Render a quadrotor flying a TUM trajectory over a carpet floor (Blender, Cycles CPU).

blender -b -P render_drone.py -- --gt gt.tum --pred pred.tum --texture carpet.jpg \
        --out out.mp4 --title "Step K | ellipse 0.5 m/s" --subtitle "ATE 0.21 m"
Pink trail = ground truth (the drone flies this). Blue trail + ghost = network prediction.
"""
import sys, math, argparse
import numpy as np
import bpy
from mathutils import Vector, Quaternion

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--gt", required=True); ap.add_argument("--pred", required=True)
ap.add_argument("--texture", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--title", default=""); ap.add_argument("--subtitle", default="")
ap.add_argument("--fps", type=int, default=24); ap.add_argument("--speedup", type=float, default=2.0)
ap.add_argument("--samples", type=int, default=96)
ap.add_argument("--res", type=int, nargs=2, default=[1280, 720])
ap.add_argument("--max_frames", type=int, default=0)
a = ap.parse_args(argv)

gt = np.loadtxt(a.gt); pr = np.loadtxt(a.pred)
T = gt[-1, 0] - gt[0, 0]
nfr = int(T / a.speedup * a.fps)
if a.max_frames: nfr = min(nfr, a.max_frames)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.samples = a.samples; sc.cycles.use_denoising = True; sc.cycles.device = "CPU"
try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for dt in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = dt; prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == dt]
            if gpus:
                for d in prefs.devices: d.use = (d.type == dt)
                sc.cycles.device = "GPU"; print("USING GPU", dt, [d.name for d in gpus]); break
        except Exception as e: print("gpu fail", dt, e)
except Exception as e: print("no gpu prefs", e)
sc.view_settings.view_transform = "AgX"
sc.render.resolution_x, sc.render.resolution_y = a.res
sc.render.fps = a.fps; sc.frame_start = 1; sc.frame_end = nfr
sc.render.image_settings.file_format = "FFMPEG"
sc.render.ffmpeg.format = "MPEG4"; sc.render.ffmpeg.codec = "H264"
sc.render.ffmpeg.constant_rate_factor = "MEDIUM"
sc.render.filepath = a.out

# --- world / lights
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes["Background"]; bg.inputs[0].default_value = (0.55, 0.68, 0.85, 1); bg.inputs[1].default_value = 0.9
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 4.0; sun.rotation_euler = (math.radians(40), math.radians(10), math.radians(35))
sc.collection.objects.link(sun)

def mat(name, rgb, rough=0.5, metal=0.0, emit=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough; b.inputs["Metallic"].default_value = metal
    if emit:
        b.inputs["Emission Color"].default_value = (*rgb, 1); b.inputs["Emission Strength"].default_value = emit
    return m

# --- carpet floor
xs = np.r_[gt[:, 1], pr[:, 1]]; ys = np.r_[gt[:, 2], pr[:, 2]]
cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
ext = max(xs.max() - xs.min(), ys.max() - ys.min()) + 3
bpy.ops.mesh.primitive_plane_add(size=1, location=(cx, cy, 0)); fl = bpy.context.object
fl.scale = (ext * 25, ext * 25, 1)
fm = bpy.data.materials.new("carpet"); fm.use_nodes = True
nt = fm.node_tree; bsdf = nt.nodes["Principled BSDF"]
tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = bpy.data.images.load(a.texture)
mp = nt.nodes.new("ShaderNodeMapping"); tc = nt.nodes.new("ShaderNodeTexCoord")
reps = ext * 25 / 8.0; mp.inputs["Scale"].default_value = (reps, reps, 1)
nt.links.new(tc.outputs["UV"], mp.inputs["Vector"]); nt.links.new(mp.outputs["Vector"], tex.inputs["Vector"])
tex.extension = "MIRROR"
hs = nt.nodes.new("ShaderNodeHueSaturation"); hs.inputs["Value"].default_value = 0.45; hs.inputs["Saturation"].default_value = 0.8
nt.links.new(tex.outputs["Color"], hs.inputs["Color"]); nt.links.new(hs.outputs["Color"], bsdf.inputs["Base Color"]); bsdf.inputs["Roughness"].default_value = 0.95
fl.data.materials.append(fm)

# --- drone model
def drone(name, body_rgb, alpha=1.0):
    parts = []
    m_body = mat(name + "b", body_rgb, 0.35, 0.2); m_dark = mat(name + "d", (0.03, 0.03, 0.03), 0.6)
    m_led = mat(name + "l", (1, 0.2, 0.1), 0.3, 0, 6.0)
    def add(op, mt, **kw):
        op(**kw); o = bpy.context.object; o.data.materials.append(mt); parts.append(o); return o
    add(bpy.ops.mesh.primitive_cube_add, m_body, size=1).scale = (0.22, 0.14, 0.05)
    add(bpy.ops.mesh.primitive_uv_sphere_add, m_led, radius=0.035, location=(0.13, 0, 0.0))
    props = []
    for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        arm = add(bpy.ops.mesh.primitive_cylinder_add, m_dark, radius=0.012, depth=0.5)
        arm.rotation_euler = (0, math.pi / 2, math.atan2(sy, sx)); arm.location = (sx * .15, sy * .15, 0)
        p = add(bpy.ops.mesh.primitive_cylinder_add, m_dark, radius=0.09, depth=0.006, location=(sx * .3, sy * .3, 0.04))
        props.append(p)
    root = bpy.data.objects.new(name, None); sc.collection.objects.link(root)
    for o in parts: o.parent = root
    root.rotation_mode = "QUATERNION"
    return root, props
d_gt, props = drone("drone", (0.95, 0.95, 0.97))
d_pr, props_pr = drone("ghost", (0.15, 0.45, 1.0))
for o in d_pr.children:  # make ghost semi-transparent look: shrink a bit
    pass
d_gt.scale = (2.2, 2.2, 2.2); d_pr.scale = (1.8, 1.8, 1.8)

# --- trails
def trail(name, arr, rgb):
    pts = arr[::5, 1:4]
    cu = bpy.data.curves.new(name, "CURVE"); cu.dimensions = "3D"
    sp = cu.splines.new("POLY"); sp.points.add(len(pts) - 1)
    for p, q in zip(sp.points, pts): p.co = (q[0], q[1], q[2], 1)
    cu.bevel_depth = 0.05; cu.bevel_resolution = 2; cu.use_fill_caps = True
    ob = bpy.data.objects.new(name, cu); sc.collection.objects.link(ob)
    ob.data.materials.append(mat(name, rgb, 0.4, 0, 2.5)); return cu
c_gt = trail("gt_trail", gt, (1.0, 0.15, 0.55)); c_pr = trail("pred_trail", pr, (0.1, 0.4, 1.0))

# --- cameras: wide / chase / top-down, switched with timeline markers
zc = np.r_[gt[:, 3], pr[:, 3]].mean()
tgt = Vector((cx, cy, zc * 0.55)); dist = ext * 0.95 + 1
def mkcam(name, loc, lens=28, ortho=False):
    c = bpy.data.objects.new(name, bpy.data.cameras.new(name)); sc.collection.objects.link(c)
    c.data.lens = lens; c.location = loc; return c
def aim(c, target):
    c.rotation_euler = (Vector(target) - c.location).to_track_quat("-Z", "Y").to_euler()
cam_wide = mkcam("wide", tgt + Vector((dist * 0.75, -dist * 0.85, dist * 0.55))); aim(cam_wide, tgt)
cam_top = mkcam("top", Vector((cx, cy, ext * 0.85 + zc)), lens=24); cam_top.rotation_euler = (0, 0, 0)
follow = bpy.data.objects.new("follow", None); sc.collection.objects.link(follow)
cam_chase = mkcam("chase", (0, 0, 0), lens=35); cam_chase.parent = follow; cam_chase.location = (-2.6, -2.6, 1.3)
tr = cam_chase.constraints.new("TRACK_TO"); tr.target = d_gt; tr.track_axis = "TRACK_NEGATIVE_Z"; tr.up_axis = "UP_Y"
sc.camera = cam_wide

def label(cam, txt, y, size, rgb=(1, 1, 1)):
    t = bpy.data.curves.new("t", "FONT"); t.body = txt; t.size = size; t.align_x = "LEFT"
    o = bpy.data.objects.new("t", t); sc.collection.objects.link(o); o.parent = cam
    o.location = (-1.9 * 1.0, y, -5.0)
    o.data.materials.append(mat("tx", rgb, 0.5, 0, 3.0)); return o
b1, b2 = int(nfr * 0.40), int(nfr * 0.75)
for fr, c in ((1, cam_wide), (b1, cam_chase), (b2, cam_top)):
    m = sc.timeline_markers.new(c.name, frame=fr); m.camera = c

# --- animation
for f in range(1, nfr + 1):
    t = (f - 1) / a.fps * a.speedup; i = min(int(round(t * 100)), len(gt) - 1)
    for obj, arr in ((d_gt, gt), (d_pr, pr)):
        r = arr[i]; obj.location = Vector(r[1:4])
        obj.rotation_quaternion = Quaternion((r[7], r[4], r[5], r[6]))
        obj.keyframe_insert("location", frame=f); obj.keyframe_insert("rotation_quaternion", frame=f)
    follow.location = Vector(gt[i, 1:4]); follow.keyframe_insert("location", frame=f)
    for p in props + props_pr:
        p.rotation_euler = (0, 0, f * 1.7); p.keyframe_insert("rotation_euler", frame=f)
    frac = i / (len(gt) - 1)
    for cu in (c_gt, c_pr):
        cu.bevel_factor_end = frac; cu.keyframe_insert("bevel_factor_end", frame=f)
for fc in [fc for ob in (d_gt, d_pr) for fc in ob.animation_data.action.fcurves]:
    for k in fc.keyframe_points: k.interpolation = "LINEAR"

bpy.ops.render.render(animation=True)
print("DONE", a.out, nfr, "frames")
