
import glob
import json
import os

import numpy as np

from utils import colmap_utils


def ids_from_fname(fn):
    splits = fn.split('/')[-2:]
    cam_id, time_id = int(splits[0]), int(splits[-1].split('.')[0])
    return cam_id, time_id

def colmap_name_from_ids(cam_id, time_id):
    return f'frame_{cam_id}_{time_id:06d}.jpg'

def dyn_name_from_ids(cam_id, time_id):
    return f'{cam_id}/{time_id:06d}.png'

def link_images(path2imgs, path2links):
    os.makedirs(path2links, exist_ok=True)

    list_files = glob.glob(os.path.join(path2imgs, '**/*.png'))
    if len(list_files) == 0:
        list_files = glob.glob(os.path.join(path2imgs, '**/*.jpg'))
    else:
        assert True, "Only png and jpg files are supported"
    for file in list_files:
        c_id, t_id = ids_from_fname(file)
        out_name = colmap_name_from_ids(c_id, t_id)
        # os.symlink(os.path.join(path2imgs, file), os.path.join(path2links, out_name))
        symlink_path = os.path.join(path2links, out_name)
        os.symlink(os.path.join(path2imgs, file), symlink_path + 'tmp')
        os.rename(symlink_path + 'tmp', symlink_path)

def read_images_json(path, n_time_steps=None):
    """
    (Q, T) -> w2c
    The reconstructed pose of an image is specified as the projection
    from world to the camera coordinate system of an image
    using a quaternion (QW, QX, QY, QZ) and a translation vector (TX, TY, TZ).
    """
    md = json.load(open(path, 'r'))  # metadata

    images = {}
    a = 0
    for i, fns in enumerate(md['fn'][:n_time_steps]):
        for j, fn in enumerate(fns):
            c_id, t_id = ids_from_fname(fn)
            image_name = fn
            image_id = c_id * 1000000 + t_id
            camera_id = md['cam_id'][i][j]

            # c2w = np.linalg.inv(md['w2c'][i][j])
            w2c = np.array(md['w2c'][i][j])
            qvec = colmap_utils.rotmat2qvec(w2c[:3, :3])
            tvec = w2c[:3, 3]


            images[image_id] = colmap_utils.Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=np.empty((0, 2), dtype=np.float32),
                point3D_ids=np.empty(0, dtype=np.int32),
            )
            a = a + 1
    return images

def read_cameras_json(path):
    # k: List[List[np.ndarray]]   #: List of list of camera intrinsics
    # cam_id: List[List[int]]     #: List of list of camera ids
    # w: int                      #: image width
    # h: int                      #: image height

    md = json.load(open(path, 'r'))  # metadata

    cameras = {}
    # Get info for each camera only from the first frame
    w, h = md['w'], md['h']
    for (k_i, cam_id_i) in zip(md['k'][0], md['cam_id'][0]):
        fx, fy, cx, cy = k_i[0][0], k_i[1][1], k_i[0][2], k_i[1][2]

        simple_pinhole = False #  (fx - fy) <= 1e-6

        cameras[cam_id_i] = colmap_utils.Camera(
            id=cam_id_i,
            model='SIMPLE_PINHOLE' if simple_pinhole else 'PINHOLE',
            width=w,
            height=h,
            params=[fx, cx, cy] if simple_pinhole else [fx, fy, cx, cy],
        )
    return cameras



def read_points3D_npz(path):
    pcd = np.load(path)["data"] # (N, 7): rgb, xyz, seg

    points3D = {}
    for i, p in enumerate(pcd):
        points3D[i] = colmap_utils.Point3D(
            id=i,
            xyz=p[:3],
            rgb=(p[3:6]).astype(np.uint8),
            error=0,
            image_ids=np.empty(0, dtype=np.int32),
            point2D_idxs=np.empty(0, dtype=np.int32),
        )
    return points3D


def generate_colmap(path_dyn_3dgs, output_path=None):
    print(f"[COLMAP]: Generating colmap model from {path_dyn_3dgs}")

    # Create folders
    output_path = output_path if output_path else path_dyn_3dgs + '/colmap'
    sparse_path = os.path.join(output_path, 'sparse/0/')
    os.makedirs(sparse_path, exist_ok=True)

    # Create cameras.txt
    cameras = read_cameras_json(os.path.join(path_dyn_3dgs, 'test_meta.json'))
    cameras.update(read_cameras_json(os.path.join(path_dyn_3dgs, 'train_meta.json')))
    cameras = dict(sorted(cameras.items(), key=lambda x: x[1].id))

    # Create images.txt
    images = read_images_json(os.path.join(path_dyn_3dgs, 'test_meta.json'), n_time_steps=1)
    images.update(read_images_json(os.path.join(path_dyn_3dgs, 'train_meta.json'), n_time_steps=1))

    # Create points3D.txt
    points3D = read_points3D_npz(os.path.join(path_dyn_3dgs, 'init_pt_cld.npz'))

    colmap_utils.write_model(cameras, images, points3D, sparse_path, ext=".txt")


    # Create links to images', depths' and segs' folders
    images_path = os.path.abspath(os.path.join(path_dyn_3dgs, 'ims'))
    depths_path = os.path.abspath(os.path.join(path_dyn_3dgs, 'depth'))
    segs_path =   os.path.abspath(os.path.join(path_dyn_3dgs, 'seg'))

    link_images(images_path, os.path.join(output_path, 'input'))
    # link_images(depths_path, os.path.join(output_path, 'depth'))
    # link_images(segs_path, os.path.join(output_path, 'seg'))
    pass


if __name__ == "__main__":
    generate_colmap("/is/sg2/mzhobro/AL_projects/0_mikel_projects/AL_3dgs/data/basketball")
