from __future__ import print_function, absolute_import, division

import os
import glob
import json
import pickle
import math
import random
from pathlib2 import Path

import numpy as np
import trimesh
import h5py

from grasp_data_reader import *


class AcronymGrasps:
    def __init__(self, filename, data_dir=None):
        self.data_dir = data_dir

        scale = None
        if filename.endswith(".json"):
            data = json.load(open(filename, "r"))
            self.mesh_fname = data["object"].decode("utf-8")
            self.mesh_type = self.mesh_fname.split("/")[1]
            self.mesh_id = self.mesh_fname.split("/")[-1].split(".")[0]
            self.mesh_scale = data["object_scale"] if scale is None else scale
        elif filename.endswith(".h5"):
            data = h5py.File(filename, "r")
            self.mesh_fname = data["object/file"][()].decode("utf-8")
            self.mesh_type = self.mesh_fname.split("/")[1]
            self.mesh_id = self.mesh_fname.split("/")[-1].split(".")[0]
            self.mesh_scale = data["object/scale"][()] if scale is None else scale
        else:
            raise RuntimeError("Unknown file ending:", filename)
        self.mesh_filepath = os.path.join(self.data_dir, self.mesh_fname)

        self.grasps, self.success = self.load_grasps(filename)
        good_idxs = np.argwhere(self.success == 1)[:, 0]
        bad_idxs = np.argwhere(self.success == 0)[:, 0]
        self.good_grasps = self.grasps[good_idxs, ...]
        self.bad_grasps = self.grasps[bad_idxs, ...]

    def load_grasps(self, filename):
        """Load transformations and qualities of grasps from a JSON file from the dataset.

        Args:
            filename (str): HDF5 or JSON file name.

        Returns:
            np.ndarray: Homogenous matrices describing the grasp poses. 2000 x 4 x 4.
            np.ndarray: List of binary values indicating grasp success in simulation.
        """
        if filename.endswith(".json"):
            data = json.load(open(filename, "r"))
            T = np.array(data["transforms"])
            success = np.array(data["quality_flex_object_in_gripper"])
        elif filename.endswith(".h5"):
            data = h5py.File(filename, "r")
            T = np.array(data["grasps/transforms"])
            success = np.array(data["grasps/qualities/flex/object_in_gripper"])
        else:
            raise RuntimeError("Unknown file ending:", filename)
        return T, success

    def load_mesh(self):
        mesh_path_file = self.mesh_filepath
        mesh = trimesh.load(mesh_path_file, file_type="obj", force="mesh")

        mesh.apply_scale(self.mesh_scale)
        if type(mesh) == trimesh.scene.scene.Scene:
            mesh = trimesh.util.concatenate(mesh.dump())
        return mesh

    def sampled_pc_filename_from_mesh_fname(self):
        mesh_filepath = Path(self.mesh_filepath)
        obj_instance = mesh_filepath.stem
        obj_class = mesh_filepath.parent.stem

        mesh_sampled_pc_dir = Path(self.data_dir) / "mesh_sampled_pc"
        pc_filename = "{0}_{1}.npy".format(obj_instance, self.mesh_scale)
        pc_filepath = mesh_sampled_pc_dir / obj_class / pc_filename
        return pc_filepath

    def write_mesh_sampled_pc(self, n_points=5000):
        mesh = self.load_mesh()
        pc = mesh.sample(n_points)

        npy_filepath = self.sampled_pc_filename_from_mesh_fname()
        if not npy_filepath.parent.exists():
            if not npy_filepath.parent.parent.exists():
                npy_filepath.parent.parent.mkdir(parents=False, exist_ok=True)
            npy_filepath.parent.mkdir(parents=False, exist_ok=True)
        print("Saving sampled point cloud {0} to: {1}".format(pc.shape, npy_filepath))
        np.save(npy_filepath, pc)

    def sdf_filename_from_mesh_fname(self):
        mesh_fname = self.mesh_fname
        mesh_type = mesh_fname.split("/")[1]
        mesh_name = mesh_fname.split("/")[-1]
        filename = mesh_name.split(".obj")[0]
        sdf_filepath = Path(self.data_dir) / "sdf" / mesh_type / (filename + ".json")
        return sdf_filepath

    def write_sdf_npy_from_json_file(self):
        mesh_scale = self.mesh_scale
        sdf_filepath = self.sdf_filename_from_mesh_fname()
        with open(sdf_filepath, "rb") as f:
            sdf_dict = pickle.load(f)

        loc = sdf_dict["loc"]
        scale = sdf_dict["scale"]
        xyz = (sdf_dict["xyz"] + loc) * scale * mesh_scale
        sdf = sdf_dict["sdf"] * scale * mesh_scale

        xyz_sdf = np.hstack([xyz, sdf.reshape(-1, 1)]).astype(np.float32)
        npy_filepath = sdf_filepath.parent / "{0}_{1}.npy".format(
            sdf_filepath.stem, mesh_scale
        )
        print(
            "Saving SDF coords,values {0} to: {1}".format(xyz_sdf.shape, npy_filepath)
        )
        np.save(npy_filepath, xyz_sdf)


def get_grasp_files(data_dir, class_type, filter_good_grasps=True):
    grasps_dir = os.path.join(data_dir, "grasps")
    print("> Searching Grasps in directory:", grasps_dir)
    grasp_files = []
    for class_type_i in class_type:
        cls_grasps_files = sorted(glob.glob(grasps_dir + "/" + class_type_i + "/*.h5"))

        for grasp_file in cls_grasps_files:
            g_obj = AcronymGrasps(grasp_file, data_dir=data_dir)

            ## Grasp File ##
            if filter_good_grasps and g_obj.good_grasps.shape[0] > 0:
                grasp_files.append(grasp_file)
            elif not filter_good_grasps:
                grasp_files.append(grasp_file)
    return grasp_files


# ---------------------------- data loader ----------------------------


class PointCloudReaderAcr:
    def __init__(
        self,
        root_folder,
        batch_size,
        num_grasp_clusters,
        npoints,
        min_difference_allowed=(0, 0, 0),
        max_difference_allowed=(3, 3, 0),
        occlusion_nclusters=0,
        occlusion_dropout_rate=0.0,
        caching=True,
        run_in_another_process=True,
        collision_hard_neg_min_translation=(-0.03, -0.03, -0.03),
        collision_hard_neg_max_translation=(0.03, 0.03, 0.03),
        collision_hard_neg_min_rotation=(-0.6, -0.2, -0.6),
        collision_hard_neg_max_rotation=(+0.6, +0.2, +0.6),
        collision_hard_neg_num_perturbations=10,
        use_uniform_quaternions=False,
        ratio_of_grasps_used=1.0,
        ratio_positive=0.3,
        ratio_hardnegative=0.4,
        balanced_data=True,
        single_view=True,
    ):
        self._root_folder = root_folder
        self._batch_size = batch_size
        self._num_grasp_clusters = num_grasp_clusters
        self._max_difference_allowed = max_difference_allowed
        self._min_difference_allowed = min_difference_allowed
        self._npoints = npoints
        self._occlusion_nclusters = occlusion_nclusters
        self._occlusion_dropout_rate = occlusion_dropout_rate
        self._caching = caching
        self._collision_hard_neg_min_translation = collision_hard_neg_min_translation
        self._collision_hard_neg_max_translation = collision_hard_neg_max_translation
        self._collision_hard_neg_min_rotation = collision_hard_neg_min_rotation
        self._collision_hard_neg_max_rotation = collision_hard_neg_max_rotation
        self._collision_hard_neg_num_perturbations = (
            collision_hard_neg_num_perturbations
        )
        self._collision_hard_neg_queue = {}
        self._ratio_of_grasps_used = ratio_of_grasps_used
        self._ratio_positive = ratio_positive
        self._ratio_hardnegative = ratio_hardnegative
        self._balanced_data = balanced_data
        self._single_view = single_view

        for i in range(3):
            assert (
                collision_hard_neg_min_rotation[i] <= collision_hard_neg_max_rotation[i]
            )
            assert (
                collision_hard_neg_min_translation[i]
                <= collision_hard_neg_max_translation[i]
            )

        self._current_pc = None
        self._cache = {}
        if run_in_another_process:
            self._renderer = OnlineObjectRendererMultiProcess(caching=True)
        else:
            self._renderer = OnlineObjectRenderer(caching=True)

        self._renderer.start()

        if use_uniform_quaternions:
            quaternions = [
                l[:-1].split("\t")
                for l in open("uniform_quaternions/data2_4608.qua", "r").readlines()
            ]

            quaternions = [
                [float(t[0]), float(t[1]), float(t[2]), float(t[3])]
                for t in quaternions
            ]
            quaternions = np.asarray(quaternions)
            quaternions = np.roll(quaternions, 1, axis=1)
            self._all_poses = [tra.quaternion_matrix(q) for q in quaternions]
        else:
            self._all_poses = []
            for az in np.linspace(0, np.pi * 2, 30):
                for el in np.linspace(-np.pi / 2, np.pi / 2, 30):
                    self._all_poses.append(tra.euler_matrix(el, az, 0))

        self._eval_files = [
            json.load(open(f))
            for f in glob.glob(os.path.join(self._root_folder, "splits", "*.json"))
        ]

    def apply_dropout(self, pc):
        if self._occlusion_nclusters == 0 or self._occlusion_dropout_rate == 0.0:
            return np.copy(pc)

        labels = farthest_points(
            pc, self._occlusion_nclusters, distance_by_translation_point
        )

        removed_labels = np.unique(labels)
        removed_labels = removed_labels[
            np.random.rand(removed_labels.shape[0]) < self._occlusion_dropout_rate
        ]
        if removed_labels.shape[0] == 0:
            return np.copy(pc)
        mask = np.ones(labels.shape, labels.dtype)
        for l in removed_labels:
            mask = np.logical_and(mask, labels != l)
        return pc[mask]

    def render_random_scene(self, camera_pose=None):
        """
        Renders a random view and return (pc, camera_pose, object_pose).
        object_pose is None for single object per scene.
        """
        if camera_pose is None:
            viewing_index = np.random.randint(0, high=len(self._all_poses))
            camera_pose = self._all_poses[viewing_index]

        in_camera_pose = copy.deepcopy(camera_pose)
        _, _, pc, camera_pose = self._renderer.render(in_camera_pose)
        pc = self.apply_dropout(pc)
        pc = regularize_pc_point_count(pc, self._npoints)
        pc_mean = np.mean(pc, 0, keepdims=True)
        pc[:, :3] -= pc_mean[:, :3]
        camera_pose[:3, 3] -= pc_mean[0, :3]

        return pc, camera_pose, in_camera_pose

    def change_object(self, cad_path, cad_scale):
        self._renderer.change_object(cad_path, cad_scale)

    def get_evaluator_data(self, grasp_path, verify_grasps=False):
        raise NotImplementedError("TODO: get_evaluator_data")
        # if self._balanced_data:
        #     return self._get_uniform_evaluator_data(grasp_path, verify_grasps)

        # (
        #     pos_grasps,
        #     pos_qualities,
        #     neg_grasps,
        #     neg_qualities,
        #     obj_mesh,
        #     cad_path,
        #     cad_scale,
        # ) = self.read_grasp_file(grasp_path)

        # output_pcs = []
        # output_grasps = []
        # output_qualities = []
        # output_labels = []
        # output_pc_poses = []
        # output_cad_paths = [cad_path] * self._batch_size
        # output_cad_scales = np.asarray([cad_scale] * self._batch_size, np.float32)

        # num_positive = int(self._batch_size * self._ratio_positive)
        # positive_clusters = self.sample_grasp_indexes(
        #     num_positive, pos_grasps, pos_qualities
        # )
        # num_negative = self._batch_size - num_positive
        # negative_clusters = self.sample_grasp_indexes(
        #     self._batch_size - num_positive, neg_grasps, neg_qualities
        # )

        # hard_neg_candidates = []
        # # Fill in Positive Examples.
        # for positive_cluster in positive_clusters:
        #     # print(positive_cluster)
        #     selected_grasp = pos_grasps[positive_cluster[0]][positive_cluster[1]]
        #     selected_quality = pos_qualities[positive_cluster[0]][positive_cluster[1]]
        #     output_grasps.append(selected_grasp)
        #     output_qualities.append(selected_quality)
        #     output_labels.append(1)
        #     hard_neg_candidates += perturb_grasp(
        #         selected_grasp,
        #         self._collision_hard_neg_num_perturbations,
        #         self._collision_hard_neg_min_translation,
        #         self._collision_hard_neg_max_translation,
        #         self._collision_hard_neg_min_rotation,
        #         self._collision_hard_neg_max_rotation,
        #     )

        # if verify_grasps:
        #     collisions, heuristic_qualities = evaluate_grasps(output_grasps, obj_mesh)
        #     for computed_quality, expected_quality, g in zip(
        #         heuristic_qualities, output_qualities, output_grasps
        #     ):
        #         err = abs(computed_quality - expected_quality)
        #         if err > 1e-3:
        #             raise ValueError(
        #                 "Heuristic does not match with the values from data generation {}!={}".format(
        #                     computed_quality, expected_quality
        #                 )
        #             )

        # # If queue does not have enough data, fill it up with hard negative examples from the positives.
        # if (
        #     grasp_path not in self._collision_hard_neg_queue
        #     or self._collision_hard_neg_queue[grasp_path].qsize() < num_negative
        # ):
        #     if grasp_path not in self._collision_hard_neg_queue:
        #         self._collision_hard_neg_queue[grasp_path] = Queue()
        #     # hard negatives are perturbations of correct grasps.
        #     random_selector = np.random.rand()
        #     if random_selector < self._ratio_hardnegative:
        #         print("add hard neg")
        #         collisions, heuristic_qualities = evaluate_grasps(
        #             hard_neg_candidates, obj_mesh
        #         )

        #         hard_neg_mask = collisions | (heuristic_qualities < 0.001)
        #         hard_neg_indexes = np.where(hard_neg_mask)[0].tolist()
        #         np.random.shuffle(hard_neg_indexes)
        #         for index in hard_neg_indexes:
        #             self._collision_hard_neg_queue[grasp_path].put(
        #                 (hard_neg_candidates[index], -1.0)
        #             )
        #     if (
        #         random_selector >= self._ratio_hardnegative
        #         or self._collision_hard_neg_queue[grasp_path].qsize() < num_negative
        #     ):
        #         for negative_cluster in negative_clusters:
        #             selected_grasp = neg_grasps[negative_cluster[0]][
        #                 negative_cluster[1]
        #             ]
        #             selected_quality = neg_qualities[negative_cluster[0]][
        #                 negative_cluster[1]
        #             ]
        #             self._collision_hard_neg_queue[grasp_path].put(
        #                 (selected_grasp, selected_quality)
        #             )

        # # Use negative examples from queue.
        # for _ in range(num_negative):
        #     # print('qsize = ', self._collision_hard_neg_queue[file_path].qsize())
        #     grasp, quality = self._collision_hard_neg_queue[grasp_path].get()
        #     output_grasps.append(grasp)
        #     output_qualities.append(quality)
        #     output_labels.append(0)

        # self.change_object(cad_path, cad_scale)
        # for iter in range(self._batch_size):
        #     if iter > 0:
        #         output_pcs.append(np.copy(output_pcs[0]))
        #         output_pc_poses.append(np.copy(output_pc_poses[0]))
        #     else:
        #         pc, camera_pose, _ = self.render_random_scene()
        #         output_pcs.append(pc)
        #         output_pc_poses.append(inverse_transform(camera_pose))

        #     output_grasps[iter] = camera_pose.dot(output_grasps[iter])

        # output_pcs = np.asarray(output_pcs, dtype=np.float32)
        # output_grasps = np.asarray(output_grasps, dtype=np.float32)
        # output_labels = np.asarray(output_labels, dtype=np.int32)
        # output_qualities = np.asarray(output_qualities, dtype=np.float32)
        # output_pc_poses = np.asarray(output_pc_poses, dtype=np.float32)

        # return (
        #     output_pcs,
        #     output_grasps,
        #     output_labels,
        #     output_qualities,
        #     output_pc_poses,
        #     output_cad_paths,
        #     output_cad_scales,
        # )

    # def _get_uniform_evaluator_data(self, grasp_path, verify_grasps=False):
    #     (
    #         pos_grasps,
    #         pos_qualities,
    #         neg_grasps,
    #         neg_qualities,
    #         obj_mesh,
    #         cad_path,
    #         cad_scale,
    #     ) = self.read_grasp_file(grasp_path)

    #     output_pcs = []
    #     output_grasps = []
    #     output_qualities = []
    #     output_labels = []
    #     output_pc_poses = []
    #     output_cad_paths = [cad_path] * self._batch_size
    #     output_cad_scales = np.asarray([cad_scale] * self._batch_size, np.float32)

    #     num_positive = int(self._batch_size * self._ratio_positive)
    #     positive_clusters = self.sample_grasp_indexes(
    #         num_positive, pos_grasps, pos_qualities
    #     )
    #     num_hard_negative = int(self._batch_size * self._ratio_hardnegative)
    #     num_flex_negative = self._batch_size - num_positive - num_hard_negative
    #     negative_clusters = self.sample_grasp_indexes(
    #         num_flex_negative, neg_grasps, neg_qualities
    #     )
    #     # print(
    #     #    'positive = {}, hard_neg = {}, flex_neg = {}'.format(
    #     #        num_positive, num_hard_negative, num_flex_negative)
    #     # )

    #     hard_neg_candidates = []
    #     # Fill in Positive Examples.

    #     for clusters, grasps, qualities in zip(
    #         [positive_clusters, negative_clusters],
    #         [pos_grasps, neg_grasps],
    #         [pos_qualities, neg_qualities],
    #     ):
    #         for cluster in clusters:
    #             selected_grasp = grasps[cluster[0]][cluster[1]]
    #             selected_quality = qualities[cluster[0]][cluster[1]]
    #             hard_neg_candidates += perturb_grasp(
    #                 selected_grasp,
    #                 self._collision_hard_neg_num_perturbations,
    #                 self._collision_hard_neg_min_translation,
    #                 self._collision_hard_neg_max_translation,
    #                 self._collision_hard_neg_min_rotation,
    #                 self._collision_hard_neg_max_rotation,
    #             )

    #     if verify_grasps:
    #         collisions, heuristic_qualities = evaluate_grasps(output_grasps, obj_mesh)
    #         for computed_quality, expected_quality, g in zip(
    #             heuristic_qualities, output_qualities, output_grasps
    #         ):
    #             err = abs(computed_quality - expected_quality)
    #             if err > 1e-3:
    #                 raise ValueError(
    #                     "Heuristic does not match with the values from data generation {}!={}".format(
    #                         computed_quality, expected_quality
    #                     )
    #                 )

    #     # If queue does not have enough data, fill it up with hard negative examples from the positives.
    #     if (
    #         grasp_path not in self._collision_hard_neg_queue
    #         or len(self._collision_hard_neg_queue[grasp_path]) < num_hard_negative
    #     ):
    #         if grasp_path not in self._collision_hard_neg_queue:
    #             self._collision_hard_neg_queue[grasp_path] = []
    #         # hard negatives are perturbations of correct grasps.
    #         collisions, heuristic_qualities = evaluate_grasps(
    #             hard_neg_candidates, obj_mesh
    #         )

    #         hard_neg_mask = collisions | (heuristic_qualities < 0.001)
    #         hard_neg_indexes = np.where(hard_neg_mask)[0].tolist()
    #         np.random.shuffle(hard_neg_indexes)
    #         for index in hard_neg_indexes:
    #             self._collision_hard_neg_queue[grasp_path].append(
    #                 (hard_neg_candidates[index], -1.0)
    #             )
    #         random.shuffle(self._collision_hard_neg_queue[grasp_path])

    #     # Adding positive grasps
    #     for positive_cluster in positive_clusters:
    #         # print(positive_cluster)
    #         selected_grasp = pos_grasps[positive_cluster[0]][positive_cluster[1]]
    #         selected_quality = pos_qualities[positive_cluster[0]][positive_cluster[1]]
    #         output_grasps.append(selected_grasp)
    #         output_qualities.append(selected_quality)
    #         output_labels.append(1)

    #     # Adding hard neg
    #     for i in range(num_hard_negative):
    #         # print('qsize = ', self._collision_hard_neg_queue[file_path].qsize())
    #         grasp, quality = self._collision_hard_neg_queue[grasp_path][i]
    #         output_grasps.append(grasp)
    #         output_qualities.append(quality)
    #         output_labels.append(0)

    #     self._collision_hard_neg_queue[grasp_path] = self._collision_hard_neg_queue[
    #         grasp_path
    #     ][num_hard_negative:]

    #     # Adding flex neg
    #     if len(negative_clusters) != num_flex_negative:
    #         raise ValueError(
    #             "negative clusters should have the same length as num_flex_negative {} != {}".format(
    #                 len(negative_clusters), num_flex_negative
    #             )
    #         )

    #     for negative_cluster in negative_clusters:
    #         selected_grasp = neg_grasps[negative_cluster[0]][negative_cluster[1]]
    #         selected_quality = neg_qualities[negative_cluster[0]][negative_cluster[1]]
    #         output_grasps.append(selected_grasp)
    #         output_qualities.append(selected_quality)
    #         output_labels.append(0)

    #     self.change_object(cad_path, cad_scale)
    #     for iter in range(self._batch_size):
    #         if iter > 0:
    #             output_pcs.append(np.copy(output_pcs[0]))
    #             output_pc_poses.append(np.copy(output_pc_poses[0]))
    #         else:
    #             pc, camera_pose, _ = self.render_random_scene()
    #             output_pcs.append(pc)
    #             output_pc_poses.append(inverse_transform(camera_pose))

    #         output_grasps[iter] = camera_pose.dot(output_grasps[iter])

    #     output_pcs = np.asarray(output_pcs, dtype=np.float32)
    #     output_grasps = np.asarray(output_grasps, dtype=np.float32)
    #     output_labels = np.asarray(output_labels, dtype=np.int32)
    #     output_qualities = np.asarray(output_qualities, dtype=np.float32)
    #     output_pc_poses = np.asarray(output_pc_poses, dtype=np.float32)

    #     return (
    #         output_pcs,
    #         output_grasps,
    #         output_labels,
    #         output_qualities,
    #         output_pc_poses,
    #         output_cad_paths,
    #         output_cad_scales,
    #     )

    def get_vae_data(self, grasp_path):
        # TODO: read grasp
        pos_grasps, pos_qualities, _, _, object_model, cad_path, cad_scale = \
            self.read_grasp_file(grasp_path)

        output_pcs = []
        output_grasps = []
        output_pc_poses = []
        output_cad_files = [cad_path] * self._batch_size
        output_cad_scales = np.asarray([cad_scale] * self._batch_size, dtype=np.float32)
        output_qualities = []

        all_clusters = self.sample_grasp_indexes(
            self._batch_size, pos_grasps, pos_qualities
        )

        if self._single_view:
            self.change_object(cad_path, cad_scale)
        else:
            # sample full PC
            pc = object_model.sample(self._npoints).astype(np.float32)
            # homogenous coordinates
            ones = np.ones(pc.shape[0], dtype=np.float32)
            pc = np.hstack((pc, ones[:, np.newaxis]))
        
        for iter in range(self._batch_size):
            selected_grasp_index = all_clusters[iter]

            selected_grasp = pos_grasps[selected_grasp_index[0]][
                selected_grasp_index[1]
            ]
            selected_quality = pos_qualities[selected_grasp_index[0]][
                selected_grasp_index[1]
            ]
            output_qualities.append(selected_quality)

            if self._single_view:
                if iter == 0:
                    # render point cloud first time only
                    pc, camera_pose, _ = self.render_random_scene()
                    output_pcs.append(pc)
                    output_pc_poses.append(inverse_transform(camera_pose))
                else:
                    # reuse rendered point cloud
                    output_pcs.append(output_pcs[0].copy())
                    output_pc_poses.append(output_pc_poses[0].copy())
            else:
                # point cloud pre-sampled
                camera_pose = np.eye(4).astype(np.float32)
                output_pcs.append(pc)
                output_pc_poses.append(camera_pose)

            output_grasps.append(camera_pose.dot(selected_grasp))

        output_pcs = np.asarray(output_pcs, dtype=np.float32)
        output_qualities = np.asarray(output_qualities, dtype=np.float32)
        output_grasps = np.asarray(output_grasps, dtype=np.float32)
        output_pc_poses = np.asarray(output_pc_poses, dtype=np.float32)
        return (
            output_pcs,
            output_grasps,
            output_pc_poses,
            output_cad_files,
            output_cad_scales,
            output_qualities,
        )

    def sample_grasp_indexes(self, n, grasps, qualities):
        """
        Stratified sampling of the graps.
        """
        nonzero_rows = [i for i in range(len(grasps)) if len(grasps[i]) > 0]
        num_clusters = len(nonzero_rows)
        replace = n > num_clusters
        if num_clusters == 0:
            raise NoPositiveGraspsException

        grasp_rows = np.random.choice(
            range(num_clusters), size=n, replace=replace
        ).astype(np.int32)
        grasp_rows = [nonzero_rows[i] for i in grasp_rows]
        grasp_cols = []
        for grasp_row in grasp_rows:
            if len(grasps[grasp_rows]) == 0:
                raise ValueError("grasps cannot be empty")

            grasp_cols.append(np.random.randint(len(grasps[grasp_row])))

        grasp_cols = np.asarray(grasp_cols, dtype=np.int32)

        return np.vstack((grasp_rows, grasp_cols)).T

    def read_grasp_file(self, path, return_all_grasps=False):
        file_name = path
        if self._caching and file_name in self._cache:
            (
                pos_grasps,
                pos_qualities,
                neg_grasps,
                neg_qualities,
                cad,
                cad_path,
                cad_scale,
            ) = copy.deepcopy(self._cache[file_name])
            return (
                pos_grasps,
                pos_qualities,
                neg_grasps,
                neg_qualities,
                cad,
                cad_path,
                cad_scale,
            )

        (
            pos_grasps,
            pos_qualities,
            neg_grasps,
            neg_qualities,
            cad,
            cad_path,
            cad_scale,
        ) = self.read_object_grasp_data(
            path,
            ratio_of_grasps_to_be_used=self._ratio_of_grasps_used,
            return_all_grasps=return_all_grasps,
        )

        if self._caching:
            self._cache[file_name] = (
                pos_grasps,
                pos_qualities,
                neg_grasps,
                neg_qualities,
                cad,
                cad_path,
                cad_scale,
            )
            return copy.deepcopy(self._cache[file_name])

        return (
            pos_grasps,
            pos_qualities,
            neg_grasps,
            neg_qualities,
            cad,
            cad_path,
            cad_scale,
        )

    def read_object_grasp_data(
        self,
        json_path,
        quality="quality_flex_object_in_gripper",
        ratio_of_grasps_to_be_used=1.0,
        return_all_grasps=False,
    ):
        """
        Reads the grasps from the json path and loads the mesh and all the
        grasps.
        """
        num_clusters = self._num_grasp_clusters
        root_folder = self._root_folder

        if num_clusters <= 0:
            raise NoPositiveGraspsException

        # load from h5 file
        grasp_file = json_path
        g_obj = AcronymGrasps(grasp_file, data_dir=self._root_folder)

        object_model = sample.Object(g_obj.mesh_filepath)
        # exit()
        object_model.rescale(g_obj.mesh_scale)
        object_model = object_model.mesh
        object_mean = np.mean(object_model.vertices, 0, keepdims=1)
        object_model.vertices -= object_mean
        
        # grasps
        grasps = g_obj.grasps.astype(np.float32)
        flex_qualities = g_obj.success
        
        grasps[:, :3, 3] -= object_mean
        heuristic_qualities = np.ones(flex_qualities.shape)

        successful_mask = np.logical_and(
            flex_qualities > 0.01, heuristic_qualities > 0.01
        )

        positive_grasp_indexes = np.where(successful_mask)[0]
        negative_grasp_indexes = np.where(~successful_mask)[0]

        positive_grasps = grasps[positive_grasp_indexes, :, :]
        negative_grasps = grasps[negative_grasp_indexes, :, :]
        positive_qualities = heuristic_qualities[positive_grasp_indexes]
        negative_qualities = heuristic_qualities[negative_grasp_indexes]

        def cluster_grasps(grasps, qualities):
            cluster_indexes = np.asarray(
                farthest_points(grasps, num_clusters, distance_by_translation_grasp)
            )
            output_grasps = []
            output_qualities = []

            for i in range(num_clusters):
                indexes = np.where(cluster_indexes == i)[0]
                if ratio_of_grasps_to_be_used < 1:
                    num_grasps_to_choose = max(
                        1, int(ratio_of_grasps_to_be_used * float(len(indexes)))
                    )
                    if len(indexes) == 0:
                        raise NoPositiveGraspsException
                    indexes = np.random.choice(
                        indexes, size=num_grasps_to_choose, replace=False
                    )

                output_grasps.append(grasps[indexes, :, :])
                output_qualities.append(qualities[indexes])

            output_grasps = np.asarray(output_grasps)
            output_qualities = np.asarray(output_qualities)

            return output_grasps, output_qualities

        if not return_all_grasps:
            positive_grasps, positive_qualities = cluster_grasps(
                positive_grasps, positive_qualities
            )
            negative_grasps, negative_qualities = cluster_grasps(
                negative_grasps, negative_qualities
            )
            num_positive_grasps = np.sum([p.shape[0] for p in positive_grasps])
            num_negative_grasps = np.sum([p.shape[0] for p in negative_grasps])
        else:
            num_positive_grasps = positive_grasps.shape[0]
            num_negative_grasps = negative_grasps.shape[0]

        return (
            positive_grasps,
            positive_qualities,
            negative_grasps,
            negative_qualities,
            object_model,
            g_obj.mesh_filepath,
            g_obj.mesh_scale
        )

    def generate_object_set(self, split_name):
        obj_files = self._eval_files[np.random.randint(0, len(self._eval_files))][
            split_name
        ]
        return os.path.join("grasps", obj_files[np.random.randint(0, len(obj_files))])

    def arrange_objects(self, meshes):
        return np.eye(4)

    def __del__(self):
        print("********** terminating renderer **************")
        self._renderer.terminate()
        self._renderer.join()
        print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grasp data reader")
    parser.add_argument(
        "--root-folder",
        help="Root dir for data",
        type=str,
        default="unified_grasp_data",
    )
    parser.add_argument(
        "--vae-mode", help="True for vae mode", action="store_true", default=False
    )
    parser.add_argument(
        "--grasps-ratio",
        help="ratio of grasps to be used from each cluster. At least one grasp is chosen from each cluster.",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--balanced_data",
        action="store_true",
        default=False,
    )
    parser.add_argument("--allowed_category", default="", type=str)

    args = parser.parse_args()
    args.root_folder = os.path.abspath(args.root_folder)
    print("Root folder", args.root_folder)

    pcreader = PointCloudReaderAcr(
        root_folder=args.root_folder,
        batch_size=64,
        num_grasp_clusters=32,
        npoints=1024,
        ratio_of_grasps_used=args.grasps_ratio,
        balanced_data=args.balanced_data,
    )

    grasp_paths = get_grasp_files(
        data_dir=args.root_folder, class_type=["Book", "Mug"], filter_good_grasps=True
    )

    for grasp_path in grasp_paths:
        if args.vae_mode:
            (
                output_pcs,
                output_grasps,
                output_pc_poses,
                output_cad_files,
                output_cad_scales,
                output_qualities,
            ) = pcreader.get_vae_data(grasp_path)
            output_labels = None
            print(output_pcs.shape)
            exit()
        else:
            # output_pcs, output_grasps, output_labels, output_qualities, output_pc_poses, output_cad_files, output_cad_scales = pcreader.get_evaluator_data(grasp_path, verify_grasps=False)
            # TODO: WIP
            raise Exception("get_evaluator_data is not implemented yet")
