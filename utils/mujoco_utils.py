import math
import os
import time

import imageio
import mujoco
import mujoco.msh2obj
import mujoco.viewer
import numpy as np
import ray
from scipy.spatial.transform import Rotation
from tqdm import tqdm

os.environ["MUJOCO_GL"] = "egl"


def set_state(m, d, state, state_ix):
    mujoco.mj_setState(m, d, state, state_ix)
    mujoco.mj_forward(m, d)


def simulate(m, d, record_time=5.):

    paused = True

    def key_callback(keycode):
        if chr(keycode) == ' ':
            nonlocal paused
            paused = not paused

    state_ix = mujoco.mjtState.mjSTATE_QPOS + mujoco.mjtState.mjSTATE_USER # part of the state that we want to save

    simulated_state_list = []
    timestamps = []
    M = mujoco.mj_stateSize(m, state_ix)
    i=0
    subsample = 1
    progress_bar = tqdm(range(0, int(record_time/m.opt.timestep / subsample)), desc="Simulation progress")

    time_until_next_step = 0.
    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        start_time = None
        start_time = time.time()
        paused_time = time.time()
        while True and d.time < record_time:
            if not paused:
                time_paused = time.time() - paused_time
                start_time = start_time + (time_paused)

                # Step the simulation
                mujoco.mj_step(m, d)
                # Save the state required to reproduce the scene (for rendering and data generation)
                state = np.empty((M,), dtype=np.float64)
                mujoco.mj_getState(m, d, state, state_ix)
                if i % subsample == 0:
                    simulated_state_list.append(state)
                    timestamps.append(d.time)

                time_until_next_step = d.time - (time.time() - start_time)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                if i % subsample == 0:
                    progress_bar.set_postfix({
                                "sim-time": f"{d.time :.7f}",
                                "real-time": f"{time.time() - start_time :.7f}",
                                "sleeping-for": time_until_next_step,
                                "paused": time_paused,
                            })
                    progress_bar.update(1)

                paused_time = time.time()
                i += 1
            else:
                time.sleep(0.005)

            # Sync with the input from viewer
            viewer.sync()

            # capture mujoco.viewer
            if not viewer.is_running():
                print(viewer.is_running())
                print(f"[SIM]: Simulated {len(simulated_state_list)} steps timestep: {m.opt.timestep*subsample}")
                progress_bar.close()
                break


    if len(simulated_state_list) > 0:
        mujoco.mj_setState(m, d, simulated_state_list[0], state_ix)
        mujoco.mj_forward(m, d)

    return state_ix, simulated_state_list, timestamps


def simulate_offline(m, d, record_time=3.):
    state_ix = mujoco.mjtState.mjSTATE_QPOS + mujoco.mjtState.mjSTATE_USER # part of the state that we want to save
    M = mujoco.mj_stateSize(m, state_ix)

    simulated_state_list = []
    timestamps = []
    i=0
    subsample = 1
    progress_bar = tqdm(range(0, int(record_time/m.opt.timestep/subsample)), desc="Simulation progress")

    start_time = time.time()
    while True and d.time < record_time:
        # Step the simulation
        mujoco.mj_step(m, d)

        # Save the state required to reproduce the scene (for rendering and data generation)
        state = np.empty((M,), dtype=np.float64)
        mujoco.mj_getState(m, d, state, state_ix)
        if i % subsample == 0:
            simulated_state_list.append(state)
            timestamps.append(d.time)

        if i % 10 == 0:
            progress_bar.set_postfix({
                        "sim-time": f"{d.time :.7f}",
                        "real-time": f"{time.time() - start_time :.7f}",
                    })
            progress_bar.update(10)

        i += 1

    print(f"[SIM]: Simulated {len(simulated_state_list)} steps timestep: {m.opt.timestep*subsample}")
    progress_bar.close()

    # RESET DATA
    if len(simulated_state_list) > 0:
        mujoco.mj_setState(m, d, simulated_state_list[0], state_ix)
        mujoco.mj_forward(m, d)

    return state_ix, simulated_state_list, timestamps


def get_geom_pointcloud(m, d, name='ycb_object', visualize=False):
    # 1. Create sparse (colored) point cloud from initial object position
    # (n_points, 7): (x, y, z, r, g, b, seg_id)
    geom = m.geom(name)
    mesh = m.mesh(geom.dataid)

    # Get texture
    material = m.material(geom.matid)
    texture = m.texture(material.texid)
    tex_h, tex_w = texture.height[0], texture.width[0]
    tex_rgb = m.tex_rgb[texture.adr[0]: texture.adr[0]+tex_w*tex_h*3].reshape(tex_h, tex_w, 3)

    # Get vertex and texture coordinates, and colors
    v_start, v_end = mesh.vertadr[0], mesh.vertadr[0]+mesh.vertnum[0]
    tex_coord_int = np.floor(m.mesh_texcoord[v_start: v_end] * np.array((tex_w-1,tex_h-1))[None, ...]).astype(int)
    pcd_colors = tex_rgb[tex_coord_int[:,1], tex_coord_int[:,0]] / 255.

    # Get position of vertices in world frame
    init_pcd = d.geom(geom.id).xpos + (d.geom(geom.id).xmat.reshape(3,3) @ m.mesh_vert[v_start: v_end].T).T

    # Create and save point cloud
    segs = np.ones((init_pcd.shape[0], 1))
    init_pcd = np.concatenate([init_pcd, pcd_colors, segs], axis=1)

    if visualize:
        import open3d as o3d
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(init_pcd[::5,:3])
        point_cloud.colors = o3d.utility.Vector3dVector(init_pcd[::5, 3:6])
        o3d.visualization.draw_geometries([point_cloud])

    return init_pcd


def get_camera_params(m, d, r, cam_name):
    # 1. camera intrinsics
    # https://mujoco.readthedocs.io/en/stable/modeling.html#cameras
    # The above specification implies a perfect point camera with no aberrations
    w, h = r.width, r.height
    fovy = np.deg2rad(m.camera(cam_name).fovy[0])

    # https://github.com/google-deepmind/dm_control/blob/87e046bfeab1d6c1ffb40f9ee2a7459a38778c74/dm_control/mujoco/engine.py#L717
    focal_y = fov2focal(fovy, h)
    # focal_x = fov2focal(fovy, w)
    focal_x = fov2focal(fovy, w)
    focal_x = focal_y
    # f_x = f_y #* (w / h)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    camera_matrix = np.array([[focal_x, 0, cx], [0, focal_y, cy], [0, 0, 1]])

    # 2. camera extrinsics
    c2w = np.eye(4)
    c2w[:3, :3] = d.cam(cam_name).xmat.reshape(3,3)
    c2w[:3, 3] = d.cam(cam_name).xpos
    w2c = np.linalg.inv(c2w)

    # https://mujoco.readthedocs.io/en/stable/modeling.html#cameras
    # Cameras look towards the negative Z axis of the camera frame, while positive X and Y correspond to right and up in the image plane, respectively.
    # rotate camera 180 degrees around x-axis
    H_flip = np.eye(4)
    H_flip[:3, :3] = Rotation.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix()
    w2c = H_flip @ w2c

    return camera_matrix.tolist(), w2c.tolist()


def render(m, d, r, cam_name):
    r.update_scene(d, camera=cam_name)

    ## 1. Render RGB
    color = r.render()


    ## 2. Render depth
    # depth is a float array, in meters.
    r.enable_depth_rendering()
    depth = r.render()
    r.disable_depth_rendering()
    ## 3. Segmentation mask
    r.enable_segmentation_rendering()
    alseg = r.render()
    alpha, seg = alseg[..., 0], alseg[..., 1]
    r.disable_segmentation_rendering()

    seg[seg==mujoco.mjtObj.mjOBJ_GEOM] = 0 # flex floor
    seg[seg==mujoco.mjtObj.mjOBJ_FLEX] = 1 # flex bodies

    # ids = np.unique(alpha[alpha != -1])
    # names = [m.body(m.geom(i).bodyid).name for i in ids]
    # bodies = [m.body(m.geom(i).bodyid).name for i in ids]
    # ycb_geom_ids = ids[["ycb" in b for b in bodies]]
    # mask = np.isin(seg, ycb_geom_ids)

    # Mask out the background
    # color = color * mask[..., None]

    ## 4. Get camera parameters
    # camera_matrix, w2c = get_camera_params(m, d, r, cam_name)

    return color, depth,  seg#, (camera_matrix.tolist(), w2c.tolist())


def get_pointcloud_from_camera_depth(m, d, r, cam_name):
    r.update_scene(d, camera=cam_name)

    ## 1. Render RGB
    color = r.render()

    ## 2. Render depth
    # depth is a float array, in meters.
    r.enable_depth_rendering()
    depth = r.render()
    r.disable_depth_rendering()

    ## 3. Segmentation mask
    r.enable_segmentation_rendering()
    seg = r.render()[:, :, 0]
    r.disable_segmentation_rendering()

    depth[seg == -1] = np.nan
    seg = seg[seg != -1][:, None]

    # only ycb objects have non-zero seg ids
    ids = np.unique(seg)
    bodies = [m.body(m.geom(i).bodyid).name for i in ids]
    ycb_geom_ids = ids[["ycb" in b for b in bodies]]
    seg = np.isin(seg, ycb_geom_ids).astype(np.float64)


    # Intrinsics and extrinsics
    w, h = r.width, r.height
    fovy = m.camera(cam_name).fovy[0]
    f_y = 0.5 * h / np.tan(np.deg2rad(fovy)/2)
    f_x = f_y * (w / h)

    # Create arrays for x and y coordinates in normalized image coordinates
    x = (np.arange(0, w) - w/2) * w / h / f_x
    y = (np.arange(0, h) - h/2) / f_y
    z = np.ones((w*h, 1))
    X, Y = np.meshgrid(x, y)  # (h, w)
    matrix = np.vstack((X.flatten(), Y.flatten())).T  # (w*h, 2)
    matrix = np.concatenate((matrix, z), axis=1)

    # points in camera-image coordinates
    points = matrix * depth.reshape((-1, 1))
    rgb = color.reshape((-1, 3))[~np.isnan(points).any(axis=1)]
    points = points[~np.isnan(points).any(axis=1)]

    # points in world coordinates
    # 2. camera extrinsics
    c2w = np.eye(4)
    c2w[:3, :3] = d.cam(cam_name).xmat.reshape(3,3)
    c2w[:3, 3] = d.cam(cam_name).xpos
    w2c = np.linalg.inv(c2w)
    H_flip = np.eye(4)
    H_flip[:3, :3] = Rotation.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix()
    w2c = H_flip @ w2c
    c2w = np.linalg.inv(w2c)

    points = (c2w[:3,:3] @ points.T + c2w[:3, 3:4]).T
    xyz_rgb = np.concatenate((points, rgb, seg), axis=1)

    # choose only 250 points
    xyz_rgb = xyz_rgb[np.random.choice(xyz_rgb.shape[0], size=min(5000, xyz_rgb.shape[0]), replace=False)]

    return xyz_rgb


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

# ------------------------------------------ Parallel rendering ------------------------------------------
@ray.remote
def render_save(gen_model_func, state_ix, ixs, states, camera_ids, cam_names, exp_path, bar=None):
    import os

    import mujoco
    os.environ["MUJOCO_GL"] = "egl"

    model, data, renderer = gen_model_func()

    state_ix_cam_ids = []
    for state_id, state in zip(ixs, states):
        state = np.array(state)
        mujoco.mj_setState(model, data, state, state_ix)
        mujoco.mj_forward(model, data)

        cam_ids = camera_ids[state_id]
        for cam_id in cam_ids:
            cam_name = cam_names[cam_id]
            color, depth, seg = render(model, data, renderer, cam_name=cam_name)
            vertex_positions = data.flexvert_xpos.copy()
            rgba = np.concatenate([color, 255*(seg>-1).astype(np.uint8)[..., None]], axis=-1)
            imageio.imwrite(exp_path / "rgbas"   / f'{cam_id}' / f"{state_id:06d}.png", rgba)
            imageio.imwrite(exp_path / "ims"   / f'{cam_id}' /   f"{state_id:06d}.png", color)
            imageio.imwrite(exp_path / "seg"   / f'{cam_id}' /   f"{state_id:06d}.png", seg.astype(np.uint8))
            imageio.imwrite(exp_path / "depth"   / f'{cam_id}' / f"{state_id:06d}.png", (depth * 10000.).astype(np.uint16))
            np.save(exp_path / "points" / f'{cam_id}' /          f"{state_id:06d}.npy", vertex_positions)

        state_ix_cam_ids.append((state_id, cam_ids))

        if bar is not None:
          bar.update.remote(1)

    return state_ix_cam_ids


@ray.remote
def create_gif_from_folder(input_folder, duration=100, loop=0):
    import os

    from PIL import Image

    output_path = os.path.join(
        os.path.dirname(input_folder), f"output_{input_folder.stem}.gif"
    )
    image_files = sorted(
        [
            f
            for f in os.listdir(input_folder)
            if f.endswith(".png") or f.endswith(".jpg")
        ]
    )

    images = []
    for image_file in image_files[::50]:
        image_path = os.path.join(input_folder, image_file)
        image = Image.open(image_path, formats=["png", "jpg"])
        images.append(image)

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
    )