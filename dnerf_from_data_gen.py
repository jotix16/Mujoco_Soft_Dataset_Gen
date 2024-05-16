import json
import os

import numpy as np

from utils.mujoco_utils import focal2fov, fov2focal

np.printoptions(precision=4, suppress=True)


def create_json(path, output_path):
    md = json.load(open(path, 'r'))  # metadata

    # Get info for each camera only from the first frame
    w, h = md['w'], md['h']


    frames = []
    N_timesteps = len(md['cam_id'])

    for t in range(N_timesteps):
        for i, cam_id_i in enumerate(md['cam_id'][t]):
            w2c_i = np.array(md['w2c'][t][i])
            # w2c_i[:3,:3] = w2c_i[:3,:3].T
            c2w_i = np.linalg.inv(w2c_i)
            c2w_i[:3, 1:3] *= -1
            # w2c_i = np.linalg.inv(c2w_i)

            c2w_i =  [x.tolist() for x in c2w_i]

            fn_i = md['fn'][t][i]
            timestamp_i = md['timestamp'][t][i]
            k_i = md['k'][t][i]
            fx, fy, cx, cy = k_i[0][0], k_i[1][1], k_i[0][2], k_i[1][2]



            frames.append({
                'camera_id': cam_id_i,
                'width': w,
                'height': h,
                'file_path': fn_i,
                # 'intrinsic': [fx, fy, cx, cy],
                'fl_x': fx,
                'fl_y': fy,
                'cx': cx,
                'cy': cy,
                'time': timestamp_i,
                'transform_matrix': c2w_i

            })

    myjson = {
        'camera_angle_x': focal2fov(fx, w),
        'frames': frames
    }

    with open(output_path, 'w') as f:
        json.dump(myjson, f, indent=4)

def storePly(path, xyz, rgb):
    import numpy as np
    from plyfile import PlyData, PlyElement

    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def generate_dnerf(path_dyn_3dgs, output_path=None):
    print(f"[DNERF]: Generating blender dataset from {path_dyn_3dgs}")

    # Create folders
    output_path = output_path if output_path else path_dyn_3dgs + '/dnerf'
    os.makedirs(output_path, exist_ok=True)

    # Create cameras.txt
    create_json(os.path.join(path_dyn_3dgs, 'test_meta.json'), os.path.join(output_path, 'transforms_test.json'))
    create_json(os.path.join(path_dyn_3dgs, 'train_meta.json'), os.path.join(output_path, 'transforms_train.json'))

    # Create points3D.ply
    points3D_path = os.path.join(output_path, 'points3d.ply')
    # points3D = read_points3D_npz(os.path.join(path_dyn_3dgs, 'init_pt_cld.npz'))

    # storePly(points3D_path,
    #          xyz=np.array([p.xyz for p in points3D.values()]),
    #          rgb=np.array([p.rgb.astype(np.float32) for p in points3D.values()])
    #          )

    pcd = np.load(os.path.join(path_dyn_3dgs, 'init_pt_cld.npz'))["data"] # (N, 7): rgb, xyz, seg
    storePly(points3D_path,
             xyz=pcd[:, :3],
             rgb=pcd[:, 3:6]
             )

    # Link images
    images_path = os.path.abspath(os.path.join(path_dyn_3dgs, 'rgbas'))
    for d in os.listdir(images_path):
        if os.path.isdir(os.path.join(images_path, d)):

            symlink_path = os.path.join(output_path, d)
            os.symlink(os.path.join(images_path, d), symlink_path + 'tmp', target_is_directory=True)
            os.rename(symlink_path + 'tmp', symlink_path)

if __name__ == "__main__":
    generate_dnerf("/home/local_mzhobro/AL_projects/mujoco_dataset_get/datasets/mujoco/sphere_gmsh")
