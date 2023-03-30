# Copyright (c) OpenMMLab. All rights reserved.
import copy
import cv2
import random
import logging
from abc import ABCMeta, abstractmethod
import torch
import json_tricks as json
import numpy as np
from torch.utils.data import Dataset
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints
import os
from PIL import Image


logger = logging.getLogger(__name__)


class Kpt2dSviewRgbImgTopDownDataset(Dataset, metaclass=ABCMeta):
    """Base class for keypoint 2D top-down pose estimation with single-view RGB
    image as the input.

    All fashion datasets should subclass it.
    All subclasses should overwrite:
        Methods:`_get_db`, 'evaluate'

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        coco_style (bool): Whether the annotation json is coco-style.
            Default: True
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 cfg,
                 args,
                 root,
                 image_set,
                 is_train,
                 transform_stu=None,
                 transform_tea=None):

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = False
        self.joints_weight = 1

        self.db = []
        self.transform_stu = transform_stu
        self.transform_tea = transform_tea
        self.label_per_class = cfg.LABEL_PER_CLASS

        self.rf_aggre = args.rf_aggre
        self.sf_aggre = args.sf_aggre
        # generate pseudo labels with visibility
        percentage = args.percentage
        if os.path.exists('data/pseudo_labels/{}shots/pseudo_labels_train_p{}.npy'.format(self.label_per_class,
                                                                                          percentage)):
            pseudol_data = np.load('data/pseudo_labels/{}shots/pseudo_labels_train_p{}.npy'.format(self.label_per_class,
                                                                                    percentage), allow_pickle=True)
            self.pseudol_data = pseudol_data.item()
        else:
            print('Generate pseudo labels')
            pseudols_data = np.load('data/pseudo_labels/{}shots/pseudo_labels_train.npy'.format(self.label_per_class),
                                    allow_pickle=True)
            pseudols = pseudols_data.item()
            sorted_confidence = np.zeros(1)
            for k in pseudols:
                sorted_confidence = np.concatenate((sorted_confidence, pseudols[k][:, 2]), axis=0)
            sorted_confidence = np.sort(sorted_confidence)
            thre = sorted_confidence[int(percentage * sorted_confidence.shape[0])]
            for k in pseudols:
                pseudols[k][:, 2] = (pseudols[k][:, 2] > thre).astype(np.float32)
            np.save('data/pseudo_labels/{}shots/pseudo_labels_train_p{}.npy'.format(self.label_per_class, percentage),
                                                                                    pseudols)
            pseudol_data = np.load('data/pseudo_labels/{}shots/pseudo_labels_train_p{}.npy'.format(self.label_per_class,
                                                                                    percentage), allow_pickle=True)
            self.pseudol_data = pseudol_data.item()


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

    def _xywh2cs(self, x, y, w, h, padding=1.25):
        """This encodes bbox(x,y,w,h) into (center, scale)

        Args:
            x, y, w, h (float): left, top, width and height
            padding (float): bounding box padding factor

        Returns:
            center (np.ndarray[float32](2,)): center of the bbox (x, y).
            scale (np.ndarray[float32](2,)): scale of the bbox w & h.
        """
        aspect_ratio = self.image_width * 1.0 / self.image_height
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        if self.is_train and np.random.rand() < 0.3:
            center += 0.4 * (np.random.rand(2) - 0.5) * [w, h]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        # pixel std is 200.0
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        # padding to include proper amount of context
        scale = scale * padding

        return center, scale

    def _get_normalize_factor(self, gts, *args, **kwargs):
        """Get the normalize factor. generally inter-ocular distance measured
        as the Euclidean distance between the outer corners of the eyes is
        used. This function should be overrode, to measure NME.

        Args:
            gts (np.ndarray[N, K, 2]): Groundtruth keypoint location.

        Return:
            np.ndarray[N, 2]: normalized factor
        """
        return np.ones([gts.shape[0], 2], dtype=np.float32)

    @abstractmethod
    def _get_db(self):
        """Load dataset."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """Evaluate keypoint results."""
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        aspect_ratio = self.image_width * 1.0 / self.image_height
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    @staticmethod
    def _write_keypoint_results(keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def _report_metric(self,
                       res_file,
                       metrics,
                       pck_thr=0.2,
                       pckh_thr=0.7,
                       auc_nor=30):
        """Keypoint evaluation.

        Args:
            res_file (str): Json file stored prediction results.
            metrics (str | list[str]): Metric to be performed.
                Options: 'PCK', 'PCKh', 'AUC', 'EPE', 'NME'.
            pck_thr (float): PCK threshold, default as 0.2.
            pckh_thr (float): PCKh threshold, default as 0.7.
            auc_nor (float): AUC normalization factor, default as 30 pixel.

        Returns:
            List: Evaluation results for evaluation metric.
        """
        info_str = []

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.db)

        outputs = []
        gts = []
        masks = []
        box_sizes = []
        threshold_bbox = []
        threshold_head_box = []

        for pred, item in zip(preds, self.db):
            outputs.append(np.array(pred['keypoints'])[:, :-1])
            gts.append(np.array(item['joints_3d'])[:, :-1])
            masks.append((np.array(item['joints_3d_visible'])[:, 0]) > 0)
            if 'PCK' in metrics:
                bbox = np.array(item['bbox'])
                bbox_thr = np.max(bbox[2:])
                threshold_bbox.append(np.array([bbox_thr, bbox_thr]))
            if 'PCKh' in metrics:
                head_box_thr = item['head_size']
                threshold_head_box.append(
                    np.array([head_box_thr, head_box_thr]))
            box_sizes.append(item.get('box_size', 1))

        outputs = np.array(outputs)
        gts = np.array(gts)
        masks = np.array(masks)
        threshold_bbox = np.array(threshold_bbox)
        threshold_head_box = np.array(threshold_head_box)
        box_sizes = np.array(box_sizes).reshape([-1, 1])

        # if 'PCK' in metrics:
        #     _, pck, _ = keypoint_pck_accuracy(outputs, gts, masks, pck_thr,
        #                                       threshold_bbox)
        #     info_str.append(('PCK', pck))
        #
        # if 'PCKh' in metrics:
        #     _, pckh, _ = keypoint_pck_accuracy(outputs, gts, masks, pckh_thr,
        #                                        threshold_head_box)
        #     info_str.append(('PCKh', pckh))
        #
        # if 'AUC' in metrics:
        #     info_str.append(('AUC', keypoint_auc(outputs, gts, masks,
        #                                          auc_nor)))
        #
        # if 'EPE' in metrics:
        #     info_str.append(('EPE', keypoint_epe(outputs, gts, masks)))
        #
        # if 'NME' in metrics:
        #     normalize_factor = self._get_normalize_factor(
        #         gts=gts, box_sizes=box_sizes)
        #     info_str.append(
        #         ('NME', keypoint_nme(outputs, gts, masks, normalize_factor)))

        return info_str

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.db)

    def __getitem__(self, idx):
        """Get the sample given index."""
        results = copy.deepcopy(self.db[idx])
        image_file = results['image_file']
        filename = results['filename'] if 'filename' in results else ''
        imgnum = results['imgnum'] if 'imgnum' in results else ''

        data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))
        img_width = data_numpy.shape[1]

        if results['use_label'] == True:
            joints = results['joints_3d']
            joints_vis = results['joints_3d_visible']
            use_label = True

        else:
            joints = self.pseudol_data.get(idx)
            joints_vis = np.repeat(joints[:, 2:3], 3, axis=1)
            use_label = False

        c = results['center']
        s = results['scale']
        score = results['bbox_score'] if 'bbox_score' in results else 1
        bbox_id = results['bbox_id']
        r = 0
        r_ema = r

        joints_vis_ema = np.copy(joints_vis)
        data_numpy_ema = np.copy(data_numpy)
        flip = False

        data_numpy_v2 = np.copy(data_numpy)
        flip_v2 = False
        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                    and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body
            c_ema = np.copy(c)
            s_ema = s

            c_v2 = np.copy(c)
            s_v2 = s
            # Strong augmentation for student input
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0
            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1
                flip = True

            # weak augmentation for agreement check
            sf_v2 = self.sf_aggre
            rf_v2 = self.rf_aggre
            s_v2 = s_v2 * np.clip(np.random.randn() * sf_v2 + 1, 1 - sf_v2, 1 + sf_v2)
            r_v2 = np.clip(np.random.randn() * rf_v2, -rf_v2 * 2, rf_v2 * 2) \
                    if random.random() <= 0.6 else 0
            if self.flip and random.random() <= 0.5:
                data_numpy_v2 = data_numpy_v2[:, ::-1, :]
                c_v2[0] = data_numpy_v2.shape[1] - c_v2[0] - 1
                flip_v2 = True
        # Transformation for student network
        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        # Transformation for teacher network
        trans_ema = get_affine_transform(c_ema, s_ema, r_ema, self.image_size)
        input_ema = cv2.warpAffine(data_numpy_ema,
                                   trans_ema,
                                   (int(self.image_size[0]), int(self.image_size[1])),
                                   flags=cv2.INTER_LINEAR)
        # Transformation for agreement check (weak augmentation)
        trans_v2 = get_affine_transform(c_v2, s_v2, r_v2, self.image_size)
        input_v2 = cv2.warpAffine(
            data_numpy_v2,
            trans_v2,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR
        )
        warpmat1 = get_affine_transform(c, s, r, self.image_size)

        input = Image.fromarray(input)
        if self.transform_stu:
            input = self.transform_stu(input)
        if self.transform_tea:
            input_ema = self.transform_tea(input_ema)
            input_v2 = self.transform_tea(input_v2)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            'bbox_id': bbox_id,
            'flip': flip,
            'index': idx,
            'use_label': use_label,
            'joints_vis_ema': joints_vis_ema,
            'center_ema': c_ema,
            'scale_ema': s_ema,
            'img_width': img_width,
            'warpmat1': warpmat1,
            'warpmat_v2': trans_v2,
            'flip_v2': flip_v2
        }

        return input, target, target_weight, input_ema, input_v2, meta

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        kpts = sorted(kpts, key=lambda x: x[key])
        num = len(kpts)
        for i in range(num - 1, 0, -1):
            if kpts[i][key] == kpts[i - 1][key]:
                del kpts[i]

        return kpts