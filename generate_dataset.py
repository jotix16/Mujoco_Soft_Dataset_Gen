"""
This script generates data for the Dynamic3DGaussians dataset.
It generates a scene with a single object and multiple cameras.
The object is randomly placed in the scene and the cameras are placed on a sphere around the object.

The folder structure is as follows:
  data
    005_tomato_soup_can
      ims
        0
          000000.png
          000001.png
          ...
        1
          000000.png
          000001.png
          ...
        ...
      rgbas
        0
            000000.png
            000001.png
            ...
        ...
      depth
        0
          000000.png
          000001.png
          ...
        1
          000000.png
          000001.png
          ...
        ...
      seg
        0
          000000.png
          000001.png
          ...
        1
          000000.png
          000001.png
          ...
        ...
      init_pt_cld.npz
      train_meta.json
      test_meta.json
"""

"""
In addition to that, the script also generates a dnerf/colmap type dataset.
"""

import json
import math
import os
import random
from pathlib import Path

import mujoco
import mujoco.msh2obj
import mujoco.viewer
import numpy as np
import ray
from ray.experimental import tqdm_ray

from colmap_from_data_gen import generate_colmap
from dnerf_from_data_gen import generate_dnerf
from utils import mujoco_utils as mu
from utils.json_utils import MyEncoder, NoIndent

ray.init(ignore_reinit_error=True)


np.set_printoptions(linewidth=400)
os.environ["MUJOCO_GL"] = "egl"





def generate_model(
    gmsh_path,
    xyz=[0.0, 0.0, 3.1],
    euler=(0, 0, 0),
    camera_positions=None,
    w=640,
    h=360,
    with_walls=False,
    transparent_floor=False,
):

    base = """
<mujoco model="ycb_scene">
    <option gravity="0 0 -9.8" timestep="0.001"/>

    <visual>
        <rgba haze="1 1 1 1"/>
        <quality shadowsize="0"/>
    </visual>

    <extension>
        <plugin plugin="mujoco.elasticity.solid"/>
    </extension>

    <worldbody>
        <light directional="true" diffuse="1.0 1.0 1.0" specular="0. 0. 0." pos="0 0 5.0" dir="0 0 -1" castshadow="false"/>
        <light directional="true" diffuse="1.0 1.0 1.0" specular="0. 0. 0." pos="0 0 0.035" dir="0 0 1" castshadow="false"/>
        <light directional="true" diffuse="1.0 1.0 1.0" specular="0. 0. 0." pos="0 0 5.0" dir="0 0 -1" castshadow="false"/>
        <light directional="true" diffuse="1.0 1.0 1.0" specular="0. 0. 0." pos="0 5 0" dir="0 -1 0" castshadow="false"/>
        <light directional="true" diffuse="1.0 1.0 1.0" specular="0. 0. 0." pos="0 -5 0" dir="0 1 0" castshadow="false"/>
        <light directional="true" diffuse="1.0 1.0 1.0" specular="0. 0. 0." pos="5 0 0" dir= "-1 0 0" castshadow="false"/>
        <light directional="true" diffuse="1.0 1.0 1.0" specular="0. 0. 0." pos="-5 0 0" dir="1 0 0" castshadow="false"/>

        <geom name="floor" type="cylinder" size="5 .03" pos="0 0 0.0" rgba="1 1 1 {a}"/>
        {cameras}

        {my_object}

    </worldbody>
</mujoco>
    """

    # We can also wrap the deformable flxcomp in a body
    my_deformable_obj = """
        <flexcomp type="gmsh" file="{gmsh_path}" pos="{p[0]} {p[1]} {p[2]}" rgba="1 .0 .0 1" name="softbody" mass="10.4">
            <contact condim="3" solref="0.01 1" solimp=".95 .99 .0001" selfcollide="none"/>
            <edge damping="1.0"/>
            <plugin plugin="mujoco.elasticity.solid">
                <config key="poisson" value="0.2"/>
                <config key="young" value="2e3"/>
            </plugin>
        </flexcomp>
        """
    camera_base = """<camera name="fixed_cam{id}" mode="fixed" pos="{p[0]} {p[1]} {p[2]}" zaxis="{p[0]} {p[1]} {p[2]}"/>"""  # mode="targetbody"  and remove yaxis, so that the camera always looks at the object
    cameras = (
        [camera_base.format(**dict(id=i, p=p)) for i, p in enumerate(camera_positions)]
        if camera_positions is not None
        else []
    )

    xml_str = base.format(
        **dict(
            my_object=my_deformable_obj.format(gmsh_path=gmsh_path, p=xyz, o=euler),
            cameras="\n".join(cameras) if len(cameras) > 0 else "",
            a=0.0 if transparent_floor else 1.0,
        )
    )
    model = mujoco.MjModel.from_xml_string(xml_str)
    renderer = mujoco.Renderer(model, width=w, height=h)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, renderer


def fibonacci_hemisphere(samples, sphere_radius, xyz=[0, 0, 0.1], whole_sphere=True):
    # Function to generate equidistant points on a hemisphere
    # samples = samples
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # Golden angle in radians

    for i in range(samples):
        z = i / float(samples - 1)  # Range from 0 to 1
        radius = math.sqrt(1 - z * z)  # Radius at y

        theta = phi * i  # Increment

        x = math.cos(theta) * radius * sphere_radius
        y = math.sin(theta) * radius * sphere_radius

        points.append((x, y, z * sphere_radius))

    if whole_sphere:
        points = np.concatenate([points, points * np.array([1, 1, -1])])

    return points + np.asarray(xyz)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cams", type=int, default=248, help="Number of cameras")
    parser.add_argument("--radius", type=float, default=2.0, help="Number of cameras")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=360, help="Image height")
    parser.add_argument("--path", type=str, default="datasets/mujoco", help="Path to save data")
    parser.add_argument("--gmsh_path", type=str, default="/home/local_mzhobro/AL_projects/mujoco_dataset_get/assets/tetra_meshes/sphere_gmsh.msh")
    parser.add_argument("--with_walls", action="store_true", help="Add walls to the scene")
    parser.add_argument("--transparent_floor", action="store_true", help="Add transparent floor to the scene")
    parser.add_argument("--silent", action="store_true", help="Hide Mujoco viewer")

    args = parser.parse_args()
    gmsh_path = args.gmsh_path
    num_cams = args.num_cams  # (args.num_cams // 2)*2
    w, h = args.width, args.height
    xyz = (0, 0, 2.2)

    # Set up folder structure
    file_name = gmsh_path.split("/")[-1].split(".")[0] + ("_transparent" if args.transparent_floor else "")
    exp_path = Path(args.path) / file_name
    print(f"[RECORDING]:  Saving data to {exp_path}")
    for i in range(num_cams):
        (exp_path / "rgbas" / f"{i}").mkdir(parents=True, exist_ok=True)
        (exp_path / "ims" / f"{i}").mkdir(parents=True, exist_ok=True)
        (exp_path / "depth" / f"{i}").mkdir(parents=True, exist_ok=True)
        (exp_path / "seg" / f"{i}").mkdir(parents=True, exist_ok=True)
        (exp_path / "points" / f"{i}").mkdir(parents=True, exist_ok=True)

    # Set up the model
    camera_positions = fibonacci_hemisphere(samples=num_cams, sphere_radius=15.0, xyz=(0, 0, 0), whole_sphere=False)
    gen_mode_func = lambda: generate_model(
        gmsh_path=gmsh_path,
        xyz=xyz,
        euler=(0, -30, 0),
        camera_positions=camera_positions,
        w=w,
        h=h,
        transparent_floor=args.transparent_floor,
    )
    model, data, renderer = gen_mode_func()

    # Cameras
    train_cams = [f"fixed_cam{cam_id}" for cam_id in range(num_cams)]
    cam_param = {
        cam_id: mu.get_camera_params(model, data, renderer, cam_name)
        for cam_id, cam_name in enumerate(train_cams)
    }

    # -----------------------------
    # 0. Simulate without rendering
    # -----------------------------
    record_time = 2.5
    if args.silent:
        state_ix, simulated_state_list, timestamps = mu.simulate_offline(m=model, d=data, record_time=record_time)
    else:
        state_ix, simulated_state_list, timestamps = mu.simulate(m=model, d=data, record_time=record_time)

    if len(simulated_state_list) == 0:
        print("No data generated")
        exit()

    # -----------------------------
    # 1. Initial point cloud from cameras in first frame
    # -----------------------------
    mu.set_state(model, data, simulated_state_list[0], state_ix)
    init_pcd = np.concatenate(
        [mu.get_pointcloud_from_camera_depth(m=model, d=data, r=renderer, cam_name=cn) for cn in train_cams],
        axis=0,
    )
    np.savez(exp_path / "init_pt_cld.npz", data=init_pcd)  # save every second point

    # DEBUG
    # import open3d as o3d
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(init_pcd[:, :3])
    # point_cloud.colors = o3d.utility.Vector3dVector(init_pcd[:, 3:6] /255.)
    # o3d.visualization.draw_geometries([point_cloud])

    # -----------------------------
    # 2. Parallel rendering/recording with RAY
    # -----------------------------
    subsample = 8
    test_ratio = 100 // subsample
    simulated_state_list = simulated_state_list[::subsample]
    timestamps = timestamps[::subsample]
    timestep = model.opt.timestep / subsample

    print("[RENDERING]:  Rendering and saving images/depth/seg/points")
    n_splits = 10
    ixs             = np.array_split(range(len(simulated_state_list)), n_splits)
    state_splits    = np.array_split(simulated_state_list, n_splits)
    camera_ids = {
        state_ix: (list(range(num_cams)) if state_ix == 0 else [random.randint(0, num_cams - 1)])
        for state_ix in range(len(simulated_state_list))
    }

    # Render and save images in parallel using ray
    remote_tqdm = ray.remote(tqdm_ray.tqdm)
    bar = remote_tqdm.remote(total=len(simulated_state_list))
    state_cam_ids = sum(
        ray.get(
            [
                mu.render_save.remote(
                    gen_mode_func, state_ix, i, s, camera_ids, train_cams, exp_path, bar
                )
                for i, s in zip(ixs, state_splits)
            ]
        ),
        [],
    )
    state_cam_ids = sorted(state_cam_ids, key=lambda x: x[0])
    bar.close.remote()

    cam_param = {
        cam_id: mu.get_camera_params(model, data, renderer, train_cams[cam_id])
        for cam_id in range(num_cams)
    }

    # gifs
    # print("[GIF]:  Generating gifs")
    # ray.get([create_gif_from_folder.remote(exp_path / "ims" / f'{cam_id}') for cam_id in range(num_cams)])

    # -----------------------
    # 3. Generate Metadata
    # -----------------------
    #
    print("[METADATA]:  Generating metadata")
    name_geoms = {
        model.geom(i).name: (i, model.body(model.geom(i).bodyid).name)
        for i in range(model.ngeom)
    }
    with open(exp_path / "seg_info.json", "w") as fp:
        json.dump(name_geoms, fp, cls=MyEncoder, indent=4)

    # Save test.meta.json, train.meta.json
    test_state_ixs = (np.random.choice(range(len(simulated_state_list)-1), size=len(simulated_state_list)//test_ratio, replace=False) + 1).tolist()

    test_meta_data = {
        'w': w,
        'h': h,
        'timestep': timestep,
        'k':         [NoIndent([cam_param[cam_id][0] for cam_id in cam_ids])          for state_ix, cam_ids in state_cam_ids if state_ix in test_state_ixs],
        'w2c':       [NoIndent([cam_param[cam_id][1] for cam_id in cam_ids])          for state_ix, cam_ids in state_cam_ids if state_ix in test_state_ixs],
        'fn':        [NoIndent([f'{cam_id}/{state_ix:06d}' for cam_id in cam_ids])    for state_ix, cam_ids in state_cam_ids if state_ix in test_state_ixs],
        'timestamp': [NoIndent([timestamps[state_ix] for cam_id in cam_ids])          for state_ix, cam_ids in state_cam_ids if state_ix in test_state_ixs],
        'cam_id':    [NoIndent(cam_ids)                                               for state_ix, cam_ids in state_cam_ids if state_ix in test_state_ixs],
    }

    train_meta_data = {
        'w': w,
        'h': h,
        'timestep': timestep,
        'k':         [NoIndent([cam_param[cam_id][0] for cam_id in cam_ids])          for state_ix, cam_ids in state_cam_ids if state_ix not in test_state_ixs],
        'w2c':       [NoIndent([cam_param[cam_id][1] for cam_id in cam_ids])          for state_ix, cam_ids in state_cam_ids if state_ix not in test_state_ixs],
        'fn':        [NoIndent([f'{cam_id}/{state_ix:06d}' for cam_id in cam_ids])    for state_ix, cam_ids in state_cam_ids if state_ix not in test_state_ixs],
        'timestamp': [NoIndent([timestamps[state_ix] for cam_id in cam_ids])          for state_ix, cam_ids in state_cam_ids if state_ix not in test_state_ixs],
        'cam_id':    [NoIndent(cam_ids)                                               for state_ix, cam_ids in state_cam_ids if state_ix not in test_state_ixs],
    }

    with open(exp_path / "train_meta.json", "w") as fp_tr, open(exp_path / "test_meta.json", "w") as fp_te:
        json.dump(train_meta_data, fp_tr, cls=MyEncoder, indent=4)
        json.dump(test_meta_data, fp_te, cls=MyEncoder, indent=4)

    # Save colmap files
    generate_colmap(exp_path, output_path=exp_path / 'colmap')
    generate_dnerf(exp_path, output_path=exp_path / 'dnerf')
