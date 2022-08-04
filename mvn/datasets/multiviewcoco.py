
import os.path as osp
import numpy as np
import pickle
import collections
from xtcocotools.coco import COCO
from collections import defaultdict
from torch.utils.data import Dataset
import time
import cv2
import copy
import random
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset
from mvn.utils.multiview import Camera, triangulate_ransac
from mvn.utils.img import get_square_bbox, resize_image, crop_image, normalize_image, scale_bbox
from mvn.utils import volumetric

from mmpose.datasets.pipelines.top_down_transform import TopDownGenerateTarget
#serials,intrinsics, extrinsics, Distortions
class Multiview_coco(Dataset):
    def __init__(self, is_train, annfile, img_prefix, serials, train_serials, intrinsics, distortions, extrinsics,
        ori_image_shape = (1520, 2688),
        image_shape=(216, 384),
        cuboid_side=2000.0,
        scale_bbox=1.5,
        norm_image=True,
        kind="mpii",
        undistort_images=False,
        crop=False
        ):
        super(Multiview_coco, self).__init__()

        self.ori_image_shape = ori_image_shape
        self.image_shape = None if image_shape is None else tuple(image_shape)
        self.scale_bbox = scale_bbox
        self.norm_image = norm_image
        self.cuboid_side = cuboid_side
        self.kind = kind
        self.undistort_images = undistort_images
        self.crop = crop
       
        self.intrinsics = np.array(intrinsics)
        self.extrinsics = np.array(extrinsics)
        self.distortions = np.array(distortions)
       
        self.annfile = annfile
        self.img_prefix = img_prefix
        self.serials = serials
        self.train_serials = train_serials
        self.serialsIndexes = [self.serials.index(serial) for serial in  self.train_serials]
        self.dataset_name = 'Multiview_coco_h36m'
        self.num_joints = 1
        self.img_prefix = img_prefix
        self.idxLoaded =[]
        self.actual_joints = {0:'elbow'}
        self.num_keypoints = 1
        self.keypoints_3d_pred = None
        proj_matricies1 = self.intrinsics @self.extrinsics[:, 0:3, :]
        
        # self.proj_matricies = np.concatenate((proj_matricies1,row4), axis = 1)
        self.proj_matricies = proj_matricies1
        self.R, self.T, self.K , self.distortions = self.get_intrinsics(self.intrinsics, self.extrinsics, self.distortions)
        self.undistortMaps = self.getUndistortMaps()
        if is_train:
            self.times = 1
        else:
            self.times = 1
        coco_style = True
        if coco_style:
            self.coco = COCO(annfile)
            if 'categories' in self.coco.dataset:
                cats = [
                    cat['name']
                    for cat in self.coco.loadCats(self.coco.getCatIds())
                ]
                self.classes = ['__background__'] + cats
                self.num_classes = len(self.classes)
                self._class_to_ind = dict(
                    zip(self.classes, range(self.num_classes)))
                self._class_to_coco_ind = dict(
                    zip(cats, self.coco.getCatIds()))
                self._coco_ind_to_class_ind = dict(
                    (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                    for cls in self.classes[1:])
            self.img_ids = self.coco.getImgIds()
            self.num_images = len(self.img_ids)
            self.id2name, self.name2id = self._get_mapping_id_name(
                self.coco.imgs)

        self.db = self._get_db()
        self.grouping = self.get_group(self.db)
        self.group_size = len(self.grouping)
        self.u2a_mapping = {0:0}
        sigma=2
        kernel=(11, 11)
        valid_radius_factor=0.0546875
        target_type='GaussianHeatmap'
        encoding='UDP'
        unbiased_encoding=False
        self.topDownGenerateTarget = TopDownGenerateTarget(sigma, kernel, valid_radius_factor, target_type, encoding, unbiased_encoding)
        pass
        self.gt_3d, self.gt_2d = self._calc_gt_3d()
        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def getUndistortMaps(self):
        cnt  = len(self.distortions)
        maps = []
        for i in range(cnt):
            K = self.intrinsics[i] * self.image_shape[0] / self.ori_image_shape[0]
            K[2,2] = 1.0
            map1, map2 = cv2.initUndistortRectifyMap(K, self.distortions[i], None, K, self.image_shape[::-1], cv2.CV_32FC1)
            maps.append((map1, map2))
        return maps

    def get_intrinsics(self, intrinsics, extrinsics, distortions):
        cnt = len(intrinsics)
        R = np.zeros((cnt, 3, 3))
        T = np.zeros((cnt, 3))
        K =  np.array(intrinsics)
        dists = np.array(distortions)
        for i in range(cnt):
            R[i][:, :] = extrinsics[i][:3, :3]
            T[i] = extrinsics[i][:3, 3]
        return R, T, K, dists
        
    def _get_db(self):
        """Load dataset."""
        gt_db = []
        bbox_id = 0
        num_joints = self.num_joints
        for img_id in self.img_ids:

            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            objs = self.coco.loadAnns(ann_ids)

            for obj in objs:
                if max(obj['keypoints']) == 0:
                    continue
                joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

                keypoints = np.array(obj['keypoints']).reshape(-1, 3)
                joints_3d[:, :2] = keypoints[:, :2]
                assert keypoints[:, 2] == 2, 'keypoints[:, 2] must be 2'
                joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

                image_file = osp.join(self.img_prefix, self.id2name[img_id])
                # center, scale = self._xywh2cs(*obj['bbox'])
                box = obj['bbox']
                center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
                scale = ((box[2] - box[0]) / 200.0, (box[3] - box[1]) / 200.0)                
                gt_db.append({
                    'image': image_file,
                    'rotation': 0,
                    'joints_2d': keypoints.reshape(-1)[:2],
                    'joints_3d': joints_3d,
                    'joints_vis': joints_3d_visible,
                    'dataset': self.dataset_name,
                    'bbox': obj['bbox'],
                    'bbox_score': 1,
                    'bbox_id': bbox_id,
                    'center':center,
                    'scale':scale,
                    'source': 'coco',
                })
                bbox_id = bbox_id + 1
        gt_db = sorted(gt_db, key=lambda x: x['bbox_id'])

        return gt_db

    @staticmethod
    def _get_mapping_id_name(imgs):
        """
        Args:
            imgs (dict): dict of image info.

        Returns:
            tuple: Image name & id mapping dicts.

            - id2name (dict): Mapping image id to name.
            - name2id (dict): Mapping image name to id.
        """
        id2name = {}
        name2id = {}
        for image_id, image in imgs.items():
            file_name = image['file_name']
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id    

    def _get_key_str(self, filepath):
        # "file_name": "../../unmerged/dongshiling/val/color/4100/00033.jpg"
        pathlist = filepath.split('/')
        person = pathlist[-4]
        filename = osp.basename(filepath)
        cur_dir = osp.dirname(filepath)
        curview = osp.basename(cur_dir)
        # parent_dir = osp.dirname(cur_dir)
        camera_id = self.serials.index(curview)
        frameNumStr = filename.split(".")[0]
        digitPart = ''.join(filter(str.isdigit, frameNumStr))
        # suffix = ''.join(filter(str.isalpha,frameNumStr))
        # frameNum = int(digitPart)      
        key=person+digitPart
        return key, camera_id

    def get_group(self, db):
        """group images which taken at the same time"""
        grouping = {}
        nitems = len(db)
        for i in range(nitems):
            imgpath = db[i]['image']
            keystr,camera_id = self._get_key_str(imgpath)
            if keystr not in grouping:
                grouping[keystr] = [-1] * len(self.serials)
            grouping[keystr][camera_id] = i

        filtered_grouping = []
        for _, v in grouping.items():
            if np.all(np.array(v) != -1):
                filtered_grouping.append(v)

        npgroup = np.array(filtered_grouping)
        sum = np.count_nonzero(npgroup, axis=0)
        # if self.is_train:
        #     filtered_grouping = filtered_grouping[::5]
        # else:
        #     filtered_grouping = filtered_grouping[::64]

        return filtered_grouping

    def _getCameraIndex(self, filePath):
        serial = filePath.split('/')[-2]
        cameraIndex = self.serials.index(serial)
        return cameraIndex

    def _make_result(self, db_rec):
        results={}
    def _isvalid3dPoints(self, kp3d):
        assert abs(kp3d[0]) < 1000 and abs(kp3d[1]) < 1000 and kp3d[2] < 2000 and kp3d[2] > 200, 'invalid 3d points'
        return True

    def _calc_gt_3d(self):
        gt_3d = []
        gt_2d = []
        for idx in range(self.group_size):
            items = self.grouping[idx]
            points_resized = []
            points_unresized = []
            assert len(items) == len(self.serials)
            for index, item in enumerate(items):
                db_rec = copy.deepcopy(self.db[item])
                image_file = osp.join(self.img_prefix, db_rec['image'])
                cameraIndex = self._getCameraIndex(image_file)                
                assert cameraIndex == index
                keypoints = db_rec['joints_2d'].reshape(-1)
                keypoints_undistorted = cv2.undistortPoints(keypoints, self.K[cameraIndex], self.distortions[cameraIndex], None, self.K[cameraIndex])[0]                
                keypoints_undistorted1 = keypoints_undistorted * self.image_shape[0] / self.ori_image_shape[0]
                points_resized.append(keypoints_undistorted1.reshape(-1))
                points_unresized.append(keypoints_undistorted.reshape(-1))
            # time0 = time.time()
            keypoint_3d_in_base_camera, inlier_list = triangulate_ransac(self.proj_matricies, points_unresized)
            assert self._isvalid3dPoints(keypoint_3d_in_base_camera)
            gt_3d.append(keypoint_3d_in_base_camera)
            gt_2d.append(points_resized)
        return np.array(gt_3d), np.array(gt_2d)
        pass
    def __getitem__(self, idx):
        sample = defaultdict(list) # return value
        idx1 = idx % self.group_size
        input, target, weight, meta = [], [], [], []
        items = self.grouping[idx1]
        items_train = [items[i] for i in self.serialsIndexes]
        points = []
        for index, item in enumerate(items_train):
            db_rec = copy.deepcopy(self.db[item])
            image_file = osp.join(self.img_prefix, db_rec['image'])
            cameraIndex = self._getCameraIndex(image_file)
            retval_camera = Camera(self.R[cameraIndex], self.T[cameraIndex], self.K[cameraIndex], self.distortions[cameraIndex], self.serials[cameraIndex])

            image = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            assert image is not None
            if self.crop:
                # crop image
                image = crop_image(image, bbox)
                retval_camera.update_after_crop(bbox)

            if self.image_shape is not None:
                # resize
                image_shape_before_resize = image.shape[:2]
                image = resize_image(image, self.image_shape)
                retval_camera.update_after_resize(image_shape_before_resize, self.image_shape)

                sample['image_shapes_before_resize'].append(image_shape_before_resize)

            image = cv2.remap(image, self.undistortMaps[cameraIndex][0], self.undistortMaps[cameraIndex][1], cv2.INTER_LINEAR)
            keypoints = db_rec['joints_2d'].reshape(-1)
            keypoints_undistorted = cv2.undistortPoints(keypoints, self.K[cameraIndex], self.distortions[cameraIndex], None, self.K[cameraIndex])[0]
            keypoints_undistorted1 = keypoints_undistorted * self.image_shape[0] / self.ori_image_shape[0]
            keypoints_undistorted2 = self.gt_2d[idx1][cameraIndex]
            assert  np.all(keypoints_undistorted1 == keypoints_undistorted2)
            target_heatmap, target_weights = self._udp_generate_target(keypoints_undistorted1, db_rec['joints_vis'])
            if False:
                keypoints_undistorted2 = (keypoints_undistorted1 + 0.5).astype(np.int32).reshape(-1)
                assert self.ori_image_shape[0] == image_shape_before_resize[0]
                cv2.circle(image, keypoints_undistorted2, 5, (0,0, 255),1 )
                cv2.imshow("image", image)
                cv2.waitKey(0)
            if self.norm_image:
                image = normalize_image(image)


            bbox = db_rec['bbox']
            assert bbox is not None
            assert db_rec['joints_2d'] is not None

            sample['images'].append(image)
            sample['detections'].append(bbox + [1.0,]) # TODO add real confidences
            sample['cameras'].append(retval_camera)
            sample['proj_matrices'].append(retval_camera.projection)
            sample['joints_2d'].append(keypoints_undistorted1)
            sample['target_heatmap'].append(target_heatmap)
            sample['target_weights'].append(target_weights)

        for index, item in enumerate(items):
            db_rec = copy.deepcopy(self.db[item])
            points.append(db_rec['joints_2d'])
        assert len(points) == len(self.serials)
        keypoint_3d_in_base_camera, inlier_list = triangulate_ransac(self.proj_matricies, points)
        # print("ransac triangulation time:", time.time() - time0)
        keypoint_3d_in_base_camera1 = self.gt_3d[idx1]
        if not np.allclose(keypoint_3d_in_base_camera , keypoint_3d_in_base_camera1, atol= 1):
            print("notclose:", keypoint_3d_in_base_camera , keypoint_3d_in_base_camera1)

        sample['keypoints_3d'] =  np.pad(
            keypoint_3d_in_base_camera1,
            ((0,1)), 'constant', constant_values=1.0).reshape(self.num_keypoints, -1)
        sample['indexes'] = idx

        if self.keypoints_3d_pred is not None:
            sample['pred_keypoints_3d'] = self.keypoints_3d_pred[idx]

        cnt = len(sample)
        assert cnt > 5
        # sample.default_factory = None
        return sample

    def __len__(self):
        return self.group_size * self.times 

    def evaluate(self, keypoints_3d_predicted, pred_2d):
        pred = pred_2d.copy().squeeze()
        keypoints_3d_predicted = keypoints_3d_predicted.squeeze()
        headsize = self.image_shape[0] / 10.0
        threshold = 0.5

        distance = np.sqrt(np.sum((self.gt_2d - pred)**2, axis=2))
        mean2d_distance1 = np.mean(distance) 
        mean2d_distance = mean2d_distance1 /( self.image_shape[0] / self.ori_image_shape[0])
        detected = (distance <= headsize * threshold)

        joint_detection_rate = np.sum(detected, axis=0) / np.float(self.gt_2d.shape[0])

        distance3d = np.sqrt(np.sum((self.gt_3d - keypoints_3d_predicted)**2, axis=1))
        diff3d = self.gt_3d - keypoints_3d_predicted
        distance3d1 = np.linalg.norm(diff3d, axis = 1)
        mean3d_distance = np.mean(distance3d1)       

        return mean2d_distance, mean3d_distance

    def _udp_generate_target(self, joints_3d, joints_3d_visible
                             ):
        """Generate the target heatmap via 'UDP' approach. Paper ref: Huang et
        al. The Devil is in the Details: Delving into Unbiased Data Processing
        for Human Pose Estimation (CVPR 2020).

        Note:
            - num keypoints: K
            - heatmap height: H
            - heatmap width: W
            - num target channels: C
            - C = K if target_type=='GaussianHeatmap'
            - C = 3*K if target_type=='CombinedTarget'

        Args:
            cfg (dict): data config
            joints_3d (np.ndarray[K, 3]): Annotated keypoints.
            joints_3d_visible (np.ndarray[K, 3]): Visibility of keypoints.
            factor (float): kernel factor for GaussianHeatmap target or
                valid radius factor for CombinedTarget.
            target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
                GaussianHeatmap: Heatmap target with gaussian distribution.
                CombinedTarget: The combination of classification target
                (response map) and regression target (offset map).

        Returns:
            tuple: A tuple containing targets.

            - target (np.ndarray[C, H, W]): Target heatmaps.
            - target_weight (np.ndarray[K, 1]): (1: visible, 0: invisible)
        """
        num_joints = self.num_joints
        image_size = np.array(self.image_shape[::-1])
        heatmap_size = image_size // 4
        joint_weights = [1.0]
        use_different_joint_weights = False

        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_3d_visible[:, 0]

        if True:
            target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]),
                              dtype=np.float32)
            factor= 2
            tmp_size = factor * 3

            # prepare for gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, None]

            for joint_id in range(num_joints):
                feat_stride = (image_size - 1.0) / (heatmap_size - 1.0)
                mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                mu_x_ac = joints_3d[joint_id][0] / feat_stride[0]
                mu_y_ac = joints_3d[joint_id][1] / feat_stride[1]
                x0 = y0 = size // 2
                x0 += mu_x_ac - mu_x
                y0 += mu_y_ac - mu_y
                g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * factor**2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return target, target_weight
def test():
    
    from pypose.multiview.paraReader import ParaReader    
    from pypose.multiview.serialFolders import SerialFolders
    import os
    print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))    
    #annfile, img_prefix, serials
    serials = "4105,4097,4112,4102,4103,4113,4101,4114,4100,4099,4098".split(",")
    trainserials = "4114,4100,4099,4098".split(",")
    annfile = "/2t/data/recordedSamples/pose2/20220708/val.json"
    img_prefix = osp.dirname(annfile)
    cam_xmlfolder, imageResolution, imgResize = "/home/yl/working/pypose/configs/uvc", (2688, 1520), (2688, 1520)
    SerialFolders.setSerials(serials)
    paraReader = ParaReader(cam_xmlfolder, serials, imageResolution, imgResize)
    paraReader.readPara()    
    is_train = True
    coco_h36m = Multiview_coco(is_train, annfile, img_prefix, serials, trainserials, paraReader.intrinsics, paraReader.distortions, paraReader.extrinsics)
    cnt = len(coco_h36m)
    for data in coco_h36m:
        print(data['indexes'])
        
        pass
   
    pass
if __name__ == '__main__':
    test()