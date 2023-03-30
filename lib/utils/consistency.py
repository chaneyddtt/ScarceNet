
import numpy as np
import cv2
import torchvision.transforms as transforms
from core.inference import get_final_preds_const, get_final_preds
from utils.transforms import get_affine_transform, flip_back


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
trans_inp = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


def prediction_check(inp, model, dataset, c_ori, s_ori, num_transform=1, num_kpts=17):
    s0 = np.array([256/200.0, 256/200.0], dtype=np.float32)
    sf = 0.25
    rf = 30
    c = np.array([128, 128])
    image_size = np.array([256, 256])
    score_map_avg = np.zeros((1, num_kpts, 64, 64))

    for i in range(num_transform):
        img = inp.clone().numpy()
        if i == 0:
            s = s0
            r = 0
        else:
            s = s0 * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        trans = get_affine_transform(c, s, r, image_size)
        img = cv2.warpAffine(
            img,
            trans,
            (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR)

        input = (trans_inp(img)).unsqueeze(0)
        outputs, _ = model(input.cuda())
        score_map = outputs[-1] if isinstance(outputs, list) else outputs
        feat_map = score_map.squeeze(0).detach().cpu().numpy()

        flip_input = input.flip(3)
        flip_output, _ = model(flip_input.cuda())
        flip_output_re = flip_back(flip_output.detach().cpu().numpy(),
                                    dataset.flip_pairs)
        feat_map += np.squeeze(flip_output_re)
        feat_map /= 2
        M = cv2.getRotationMatrix2D((32, 32), -r, 1)
        feat_map = cv2.warpAffine(feat_map.transpose(1, 2, 0), M, (64, 64))
        feat_map = cv2.resize(feat_map, None, fx=s[0]*200.0/256.0, fy=s[1]*200.0/256.0, interpolation=cv2.INTER_LINEAR)
        if feat_map.shape[0] < 64:
            start = 32 - feat_map.shape[0]//2
            end = start + feat_map.shape[0]
            score_map_avg[0][:, start:end, start:end] += feat_map.transpose(2, 0, 1)
        else:
            start = feat_map.shape[0]//2 - 32
            end = feat_map.shape[0]//2 + 32
            score_map_avg[0] += feat_map[start:end, start:end].transpose(2, 0, 1)

    score_map_avg = score_map_avg/num_transform
    confidence_score = np.max(score_map_avg, axis=(0, 2, 3))

    confidence = confidence_score.astype(np.float32)
    preds, _ = get_final_preds_const(score_map_avg, c_ori, s_ori)
    generated_kpts = np.zeros((num_kpts, 3)).astype(np.float32)
    generated_kpts[:, :2] = preds[0, :, :2]
    generated_kpts[:, 2] = confidence
    return generated_kpts, score_map_avg


def generate_target(joints, joints_vis):
    '''
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    num_joints = 17
    image_size = np.array([256, 256])
    heatmap_size = np.array([64, 64])
    sigma = 2
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    target = np.zeros((num_joints, heatmap_size[0], heatmap_size[1]), dtype=np.float32)

    tmp_size = sigma * 3

    for joint_id in range(17):
        feat_stride = image_size / heatmap_size
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
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
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

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

    return np.expand_dims(target, axis=0), target_weight