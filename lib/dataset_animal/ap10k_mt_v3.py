# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from collections import OrderedDict, defaultdict
import logging
import sys
sys.path.insert(0, './lib')

import cv2
import json_tricks as json
import numpy as np
from xtcocotools.cocoeval import COCOeval
from xtcocotools.coco import COCO
from nms.nms import oks_nms
from nms.nms import soft_oks_nms

from dataset_animal.kpt_2d_base_mt_pseudol_v3 import Kpt2dSviewRgbImgTopDownDataset
from dataset_animal.dataset_info import DatasetInfo
from dataset_animal import ap10k_info


data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
)
logger = logging.getLogger(__name__)
ap10k2coco = np.array([2, 0, 1, 3, 4, 5, 8, 6, 9, 7, 10, 11, 14, 12, 15, 13, 16])
class AnimalAP10KDataset(Kpt2dSviewRgbImgTopDownDataset):
    """AP-10K dataset for animal pose estimation.

    `AP-10K: A Benchmark for Animal Pose Estimation in the Wild’
        Neurips Dataset Track'2021
    More details can be found in the `paper
    <https://arxiv.org/abs/2108.12617>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    AP-10K keypoint indexes::

        0: 'L_Eye',
        1: 'R_Eye',
        2: 'Nose',
        3: 'Neck',
        4: 'root of tail',
        5: 'L_Shoulder',
        6: 'L_Elbow',
        7: 'L_F_Paw',
        8: 'R_Shoulder',
        9: 'R_Elbow',
        10: 'R_F_Paw,
        11: 'L_Hip',
        12: 'L_Knee',
        13: 'L_B_Paw',
        14: 'R_Hip',
        15: 'R_Knee',
        16: 'R_B_Paw'

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
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
                 transform_tea=None
                 ):

        super().__init__(
            cfg,
            args,
            root,
            image_set,
            is_train,
            transform_stu,
            transform_tea
        )

        self.nms_thr = data_cfg['nms_thr']
        self.image_thre = data_cfg.get('det_bbox_thr', 0.0)
        self.soft_nms = data_cfg['soft_nms']
        self.oks_thre = data_cfg['oks_thr']
        self.in_vis_thre = data_cfg['vis_thr']
        self.bbox_file = data_cfg['bbox_file']
        self.use_gt_bbox = data_cfg['use_gt_bbox']

        self.use_nms = data_cfg.get('use_nms', True)

        self.image_width = data_cfg['image_size'][0]
        self.image_height = data_cfg['image_size'][1]
        self.pixel_std = 200

        self.root = root
        self.select_data = cfg.DATASET.SELECT_DATA

        if image_set == 'train':
            logger.info('Loading training annotations')
            ann_file = self.root + 'annotations/ap10k-train-split1.json'
        elif image_set == 'val':
            logger.info('Loading validation annotations')
            ann_file = self.root + 'annotations/ap10k-val-split1.json'
        else:
            logger.info('Loading testing annotations')
            ann_file = self.root + 'annotations/ap10k-test-split1.json'
        self.coco = COCO(ann_file)
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
        logger.info('=> num_images: {}'.format(self.num_images))

        self.id2name, self.name2id = self._get_mapping_id_name(
            self.coco.imgs)
        dataset_info = ap10k_info.dataset_info
        dataset_info = DatasetInfo(dataset_info)
        self.num_joints = dataset_info.keypoint_num
        self.flip_pairs = dataset_info.flip_pairs
        self.parent_ids = None
        self.upper_body_ids = dataset_info.upper_body_ids
        self.lower_body_ids = dataset_info.lower_body_ids
        self.joints_weight = np.array(dataset_info.joint_weights,
                                      dtype=np.float32).reshape((self.num_joints, 1))

        self.sigmas = dataset_info.sigmas

        try: args.few_shot_setting
        except AttributeError: self.few_shot_setting=True
        else:self.few_shot_setting=args.few_shot_setting

        # self.few_shot_setting = True
        if self.few_shot_setting and image_set == 'train':
            print('Few shot setting')
            self.annotation_per_category = cfg.LABEL_PER_CLASS
            with open('data/label_list/annotation_list_{}'.format(self.annotation_per_category), 'r') as fp:
                self.imageid_annot = json.load(fp)
            print('number of annotated images: {}'.format(len(self.imageid_annot)))
            self.img_ids_annot = self.imageid_annot

        elif cfg.DATASET.SELECT_DATA:
            print('Transfer setting')
            with open('data/label_list/annotation_list_{}'.format(cfg.DATASET.SUPERCATEGORY[0]), 'r') as fp:
                self.imageid_annot = json.load(fp)
            print('number of annotated images: {}'.format(len(self.imageid_annot)))
            self.img_ids_annot = self.imageid_annot

        else:
            raise NotImplementedError

        self.db, self.id2Cat = self._get_db()
        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        """Load dataset."""
        assert self.use_gt_bbox
        gt_db, id2Cat = self._load_coco_keypoint_annotations()
        return gt_db, id2Cat

    def _load_coco_keypoint_annotations(self):
        """Ground truth bbox and keypoints."""
        gt_db, id2Cat = [], dict()
        for img_id in self.img_ids:
            db_tmp, id2Cat_tmp = self._load_coco_keypoint_annotation_kernel(
                img_id)
            gt_db.extend(db_tmp)
            id2Cat.update({img_id: id2Cat_tmp})
        return gt_db, id2Cat

    def _supercat2ids(self):
        self.supercat2ids = dict()
        for k in self.coco.cats.keys():
            supercategory = self.coco.cats[k]['supercategory']
            id = self.coco.cats[k]['id']
            if supercategory in self.supercat2ids.keys():
                self.supercat2ids[supercategory].append(id)
            else:
                self.supercat2ids[supercategory] = [id]

    def _load_coco_keypoint_annotation_kernel(self, img_id):
        """load annotation from COCOAPI.

        Note:
            bbox:[x1, y1, w, h]
        Args:
            img_id: coco image id
        Returns:
            dict: db entry
        """
        img_ann = self.coco.loadImgs(img_id)[0]
        width = img_ann['width']
        height = img_ann['height']
        num_joints = self.num_joints

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)

        # sanitize bboxes
        valid_objs = []

        use_label = True if img_id in self.imageid_annot else False
        for obj in objs:
            if 'bbox' not in obj:
                continue
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        bbox_id = 0
        rec = []
        id2Cat = []
        for obj in objs:
            if 'keypoints' not in obj:
                continue
            if max(obj['keypoints']) == 0:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                continue
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

            keypoints = np.array(obj['keypoints']).reshape(-1, 3)
            # keypoints = keypoints[ap10k2coco]
            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

            center, scale = self._xywh2cs(*obj['clean_bbox'][:4])

            image_file = os.path.join(self.root, 'data', self.id2name[img_id])
            rec.append({
                'image_file': image_file,
                'center': center,
                'scale': scale,
                'bbox': obj['clean_bbox'][:4],
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'bbox_score': 1,
                'bbox_id': bbox_id,
                'use_label': use_label
            })
            category = obj['category_id']
            id2Cat.append({
                'image_file': image_file,
                'bbox_id': bbox_id,
                'category': category,
            })
            bbox_id = bbox_id + 1

        return rec, id2Cat

    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path,
                 *args, **kwargs):
        rank = cfg.RANK

        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            try:
                os.makedirs(res_folder)
            except Exception:
                logger.error('Fail to make {}'.format(res_folder))

        res_file = os.path.join(
            res_folder, 'keypoints_{}_results_{}.json'.format(
                self.image_set, rank)
        )

        # person x (keypoints)
        _kpts = []
        for idx, kpt in enumerate(preds):
            image_name = img_path[idx][-16:]
            image_id = self.name2id[image_name]
            bbox_id = int(all_boxes[idx][6])
            cat = self.id2Cat[image_id][bbox_id]['category']
            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': int(img_path[idx][-16:-4]),
                'bbox_id': bbox_id,
                'category': cat
            })
        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)
        kpts = self._sort_and_unique_bboxes(kpts)
        # rescoring and oks nms
        num_joints = self.num_joints
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            if self.soft_nms:
                keep = soft_oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )
            else:
                keep = oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre, self.sigmas
                )

            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file)

        info_str = self._do_python_keypoint_eval(
            res_file, res_folder)
        name_value = OrderedDict(info_str)
        return name_value, name_value['AP']

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {
                'cat_id': self._class_to_coco_ind[cls],
                'cls_ind': cls_ind,
                'cls': cls,
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if not cls == '__background__'
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array([img_kpts[k]['keypoints']
                                    for k in range(len(img_kpts))])
            key_points = np.zeros(
                (_key_points.shape[0], self.num_joints * 3), dtype=np.float
            )

            for ipt in range(self.num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            result = [
                {
                    'image_id': img_kpts[k]['image'],
                    'category_id': img_kpts[k]['category'],
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                    'center': list(img_kpts[k]['center']),
                    'scale': list(img_kpts[k]['scale'])
                }
                for k in range(len(img_kpts))
            ]
            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints', self.sigmas)
        coco_eval.params.useSegm = None
        if self.select_data:
            coco_eval.params.imgIds = self.img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        return info_str

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        for img_id, persons in kpts.items():
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts