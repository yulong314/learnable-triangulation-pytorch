
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
        crop=True
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
            self.times = 10
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
        pass

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
                # retval_camera.update_after_crop(bbox)

            if self.image_shape is not None:
                # resize
                image_shape_before_resize = image.shape[:2]
                image = resize_image(image, self.image_shape)
                # retval_camera.update_after_resize(image_shape_before_resize, self.image_shape)

                sample['image_shapes_before_resize'].append(image_shape_before_resize)

            image = cv2.remap(image, self.undistortMaps[cameraIndex][0], self.undistortMaps[cameraIndex][1], cv2.INTER_LINEAR)
            keypoints = db_rec['joints_2d'].reshape(-1)
            keypoints_undistorted = cv2.undistortPoints(keypoints, self.K[cameraIndex], self.distortions[cameraIndex], None, self.K[cameraIndex])[0]
            keypoints_undistorted1 = keypoints_undistorted * self.image_shape[0] / self.ori_image_shape[0]
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

        for index, item in enumerate(items):
            db_rec = copy.deepcopy(self.db[item])
            points.append(db_rec['joints_2d'])
        # time0 = time.time()
        keypoint_3d_in_base_camera, inlier_list = triangulate_ransac(self.proj_matricies, points)
        # print("ransac triangulation time:", time.time() - time0)
        sample['keypoints_3d'] =  np.pad(
            keypoint_3d_in_base_camera,
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

    def evaluate(self, pred, *args, **kwargs):
        pred = pred.copy()

        headsize = self.image_size[0] / 10.0
        threshold = 0.5

        u2a = self.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = list(a2u.values())
        indexes = list(range(len(a)))
        indexes.sort(key=a.__getitem__)
        sa = list(map(a.__getitem__, indexes))
        su = np.array(list(map(u.__getitem__, indexes)))

        gt = []
        for items in self.grouping:
            for item in items:
                gt.append(self.db[item]['joints_2d'][su, :2])
        gt = np.array(gt)
        pred = pred[:, su, :2]

        distance = np.sqrt(np.sum((gt - pred)**2, axis=2))
        detected = (distance <= headsize * threshold)

        joint_detection_rate = np.sum(detected, axis=0) / np.float(gt.shape[0])

        name_values = collections.OrderedDict()
        joint_names = self.actual_joints
        for i in range(len(a2u)):
            name_values[joint_names[sa[i]]] = joint_detection_rate[i]
        return name_values, np.mean(joint_detection_rate)

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