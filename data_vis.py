from __future__ import print_function, absolute_import, division
from typing import Optional

import numpy as np
import open3d as o3d
from open3d import visualization as vis
import io
from PIL import Image


def mesh_obj_colorize_o3d_(scene_mesh):
    try:
        _colors = np.array([128, 0, 128], dtype=np.float32) / 255.0
        scene_mesh.paint_uniform_color(_colors)  # Open3D supports RGB, not RGBA
        scene_mesh.compute_triangle_normals()
        scene_mesh.compute_vertex_normals()

    except Exception as e:
        print("Error in setting mesh color")


def pc_to_o3d(pc):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    # colorize
    z = pc[:, 2].reshape(-1, 1).copy()
    z = (z - z.min()) / (z.max() - z.min())
    colors = (1 - z) * np.array([255, 0, 0])[None, :] + z * np.array([0, 0, 255])[
        None, :
    ]
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    return pcd


def create_gripper_marker_o3d(
    color=[0, 0, 255], tube_radius=0.001, sections=6, scale=1.0
):
    """Create a 3D mesh visualizing a parallel yaw gripper using Open3D. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        o3d.geometry.TriangleMesh: A mesh that represents a simple parallel yaw gripper.
    """

    def create_cylinder(radius, height, resolution, translation):
        cylinder = o3d.geometry.create_mesh_cylinder(
            radius=radius, height=height, resolution=resolution
        )
        cylinder.compute_vertex_normals()
        cylinder.paint_uniform_color(np.array(color) / 255.0)
        cylinder.translate(translation)
        return cylinder

    # Create cylinders for the gripper
    cfl = create_cylinder(
        0.002 * scale,
        0.11217 * scale - 0.0659999996 * scale,
        sections,
        [0.041 * scale, 0, (0.0659999996 + 0.11217) / 2 * scale],
    )  # left clip
    cfr = create_cylinder(
        0.002 * scale,
        0.11217 * scale - 0.0659999996 * scale,
        sections,
        [-0.041 * scale, 0, (0.0659999996 + 0.11217) / 2 * scale],
    )  # right clip
    cb1 = create_cylinder(
        0.002 * scale,
        0.0659999996 * scale,
        sections,
        [0, 0, 0.0659999996 / 2 * scale],
    )  # rear handle
    cb2 = create_cylinder(
        0.002 * scale,
        0.082 * scale,
        sections,
        [0, 0, 0.0659999996 * scale],
    )  # crossbar
    cb2.rotate([0, np.pi / 2, 0], center=True)

    # cb2.translate([0, 0, 0.0659999996 * scale], relative=False)
    axes = o3d.geometry.create_mesh_coordinate_frame(size=0.02)

    # Combine all parts into a single mesh
    gripper = cfl + cfr + cb1 + cb2
    gripper = gripper + axes
    return gripper


def get_scene_geoms_o3d(
    mesh_obj,
    pc_surf,
    pc_surf_mean,
    rand_H=None,
    H_grasps_gt=None,
    H_grasps_pred=None,
    dataset_scale=1.0,
    gripper_scale=1.0,
):
    # context (pc)
    pc_surf = pc_to_o3d(pc_surf)
    # mesh (for vis)
    if mesh_obj is not None:
        mesh_obj.scale(dataset_scale, center=True)
        mesh_obj.translate(-pc_surf_mean)
        if rand_H is not None:
            mesh_obj.transform(rand_H)
        mesh_obj.translate(pc_surf_mean)
        mesh_obj.scale(1 / dataset_scale, center=True)

        mesh_obj_colorize_o3d_(mesh_obj)

    # gripper gt
    grasp_gt_list = []
    if H_grasps_gt is not None:
        for i in range(H_grasps_gt.shape[0]):
            _grasp_gt = create_gripper_marker_o3d(
                scale=gripper_scale, color=[0, 255, 0]  # green
            )
            _grasp_gt.transform(H_grasps_gt[i])
            grasp_gt_list.append({"grasp-gt-{0}".format(i): _grasp_gt})

    # gripper pred
    grasp_pred_list = []
    if H_grasps_pred is not None:
        for i in range(H_grasps_pred.shape[0]):
            # gripper
            _grasp_pred = create_gripper_marker_o3d(
                scale=gripper_scale, color=[0, 0, 255]  # blue
            )
            _grasp_pred.transform(H_grasps_pred[i])
            grasp_pred_list.append({"grasp-pred-{0}".format(i): _grasp_pred})
            # TODO: add option for grasp axes

    scene_geoms = [
        # model input:
        {"pc": pc_surf},
    ]
    scene_geoms += grasp_gt_list
    scene_geoms += grasp_pred_list
    if mesh_obj is not None:
        scene_geoms.append({"object": mesh_obj})
    return scene_geoms


def render_write_o3d(geoms, out_path, camera_params=dict()):
    # geoms is a list of key, value pairs. ref: get_scene_geoms_o3d()

    # Setup renderer
    h, w = 1080, 1920
    # w, h = 640, 480
    render = o3d.visualization.rendering.OffscreenRenderer(width=w, height=h)

    # define material (TODO: move to get_scene_geoms_o3d)
    mtrl = vis.rendering.MaterialRecord()
    mtrl.shader = "defaultLitTransparency"
    mtrl.base_roughness = 0.01
    mtrl.base_reflectance = 0.25
    mtrl.thickness = 1.0
    mtrl.transmission = 0.25

    # add geometries to scene
    for idx, geom in enumerate(geoms):
        for k, v in geom.items():
            render.scene.add_geometry(k, v, mtrl)
    render.scene.set_background([0.8, 0.8, 0.8, 1.0])

    # set the camera to fit object view
    center = np.array(render.scene.bounding_box.get_center(), dtype=np.float32)
    half = np.array(render.scene.bounding_box.get_half_extent(), dtype=np.float32)

    # re-center geometries
    for geom in geoms:
        for k, v in geom.items():
            geom_tf = render.scene.get_geometry_transform(k)
            geom_tf[:3, -1] -= center
            render.scene.set_geometry_transform(k, geom_tf)
    center = np.array([0, 0, 0], dtype=np.float32)

    low = min(center - half) * 1.25
    high = max(center + half) * 1.25

    aspect = h / w
    # s = 3
    render.scene.camera.set_projection(
        vis.rendering.Camera.Projection.Ortho,
        low,
        high,
        low * aspect,
        high * aspect,
        0.1,
        200,
    )
    # render.scene.camera.look_at([0, 0, 0], [100, 100, 100], [0, 0, 1])
    half *= 1.5
    render.scene.camera.look_at(
        center,
        center + half,
        [-1, 0, 0],
    )

    # Render and write to filepath
    img = render.render_to_image()
    o3d.io.write_image(str(out_path), img)


def vis_data_paper():
    import grasp_data_reader

    batch_size = 5
    pcreader = grasp_data_reader.PointCloudReader(
        root_folder="unified_grasp_data",
        batch_size=batch_size,
        num_grasp_clusters=32,
        npoints=1024,
        min_difference_allowed=(0, 0, 0),
        max_difference_allowed=(3, 3, 0),
        occlusion_nclusters=0,
        occlusion_dropout_rate=0.0,
        use_uniform_quaternions=0,
        ratio_of_grasps_used=1.0,
        run_in_another_process=False,  # for multiprocessing
    )

    filepath = "unified_grasp_data/grasps/cylinder_cylinder017_1.0.json"
    output = pcreader.get_vae_data(filepath)
    # print(len(output))
    # print(output[0].shape, output[1].shape, output[2].shape,
    #       output[3], output[4].shape, output[5].shape)
    # print(output[4], output[5])

    # idx = 1
    for idx in range(batch_size):
        pc = output[0][idx]
        grasps = output[1][idx]
        pc_poses = output[2][idx]
        cad_fp = output[3][idx]
        cat_scale = output[4][idx]
        grasp_quals = output[5][idx]
        
        print(pc.shape, grasps.shape, pc_poses.shape, cad_fp, cat_scale, grasp_quals)
        print(pc.dtype, grasps.dtype, pc_poses.dtype, type(cad_fp), cat_scale.dtype, grasp_quals.dtype)
        continue

        mesh = o3d.io.read_triangle_mesh(cad_fp)
        mesh.compute_vertex_normals()
        mesh_obj_colorize_o3d_(mesh)
        # TODO: transform mesh to align with point cloud
        mesh.transform(np.linalg.inv(pc_poses))

        pc_surf = pc[:, :3]
        pc_surf_mean = np.mean(pc_surf, axis=0)
        grasps = grasps.reshape(1, 4, 4)

        # get_scene_geoms_o3d
        scene_geoms = get_scene_geoms_o3d(
            mesh_obj=mesh,
            pc_surf=pc_surf,
            pc_surf_mean=pc_surf_mean,
            rand_H=None,
            H_grasps_gt=grasps,
            H_grasps_pred=None,
            dataset_scale=1.0,
            gripper_scale=1.0,
        )

        scene_geoms = [list(dict_obj.values())[0] for dict_obj in scene_geoms]
        vis.draw_geometries(scene_geoms)


def vis_data_acr():
    # TODO: open3d for py2 can't open .obj meshes from ACRONYM
    import grasp_data_reader_acr

    batch_size = 10
    pcreader = grasp_data_reader_acr.PointCloudReaderAcr(
        root_folder="data-dir",
        batch_size=batch_size,
        num_grasp_clusters=32,
        npoints=1024,
        min_difference_allowed=(0, 0, 0),
        max_difference_allowed=(3, 3, 0),
        occlusion_nclusters=0,
        occlusion_dropout_rate=0.0,
        use_uniform_quaternions=0,
        ratio_of_grasps_used=1.0,
        run_in_another_process=False,  # for multiprocessing
        single_view=False,  # full point cloud
    )

    grasp_paths = grasp_data_reader_acr.get_grasp_files(
        data_dir=pcreader._root_folder, class_type=["Book", "Mug"], filter_good_grasps=True
    )
    filepath = grasp_paths[-1]
    output = pcreader.get_vae_data(filepath)
    # print(len(output))
    # print(output[0].shape, output[1].shape, output[2].shape,
    #       output[3], output[4].shape, output[5].shape)
    # print(output[4], output[5])
    # exit()

    # idx = 1
    for idx in range(batch_size):
        pc = output[0][idx]
        grasps = output[1][idx]
        pc_poses = output[2][idx]
        cad_fp = output[3][idx]
        cat_scale = output[4][idx]
        grasp_quals = output[5][idx]
        
        print(pc.shape, grasps.shape, pc_poses.shape, cad_fp, cat_scale, grasp_quals)
        print(pc.dtype, grasps.dtype, pc_poses.dtype, type(cad_fp), cat_scale.dtype, grasp_quals.dtype)
        continue

        mesh = o3d.io.read_triangle_mesh(cad_fp)
        mesh.compute_vertex_normals()
        mesh_obj_colorize_o3d_(mesh)
        mesh.transform(np.linalg.inv(pc_poses))

        pc_surf = pc[:, :3]
        pc_surf_mean = np.mean(pc_surf, axis=0)
        grasps = grasps.reshape(1, 4, 4)

        # get_scene_geoms_o3d
        scene_geoms = get_scene_geoms_o3d(
            mesh_obj=mesh,
            pc_surf=pc_surf,
            pc_surf_mean=pc_surf_mean,
            rand_H=None,
            H_grasps_gt=grasps,
            H_grasps_pred=None,
            dataset_scale=1.0,
            gripper_scale=1.0,
        )

        scene_geoms = [list(dict_obj.values())[0] for dict_obj in scene_geoms]
        vis.draw_geometries(scene_geoms)


def main():
    vis_data_paper()
    # vis_data_acr()


if __name__ == "__main__":
    """
    for py2 (providing here to avoid the time-taking PySide build in requirements.txt):
    ```sh
    pip install tqdm h5py trimesh==3.23.3 opencv-python==4.2.0.32 python-fcl pyyaml scipy matplotlib \
        ipywidgets==7.5.0 imageio==2.6.1 pyrender==0.1.18 open3d-python
    ```
    """
    main()
