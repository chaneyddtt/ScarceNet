import cv2
import numpy as np
from torchvision.transforms import transforms


def cv2_plot_lines(frame, pts, order):
    color_mapping = {1: [255, 0, 255], 2: [255, 0, 0], 3: [255, 0, 127], 4: [255, 255, 255], 5: [0, 0, 255],
                     6: [0, 127, 255], 7: [0, 255, 255], 8: [0, 255, 0], 9: [200, 162, 200]}
    # point_size = 7
    point_size = 2
    if order == 0:
        # other animals
        # plot nose-eyes
        cv2.line(frame, (pts[2, 0], pts[2, 1]), (pts[0, 0], pts[0, 1]), color_mapping[5], point_size)
        cv2.line(frame, (pts[2, 0], pts[2, 1]), (pts[1, 0], pts[1, 1]), color_mapping[5], point_size)
        cv2.line(frame, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), color_mapping[5], point_size)

        # plot neck and nose
        # cv2.line(frame, (pts[2, 0], pts[2, 1]), (pts[3, 0], pts[3, 1]), color_mapping[8], point_size)

        # plot neck and base tail
        cv2.line(frame, (pts[4, 0], pts[4, 1]), (pts[3, 0], pts[3, 1]), color_mapping[8], point_size)

        # plot left front leg
        cv2.line(frame, (pts[3, 0], pts[3, 1]), (pts[5, 0], pts[5, 1]), color_mapping[1], point_size)
        cv2.line(frame, (pts[5, 0], pts[5, 1]), (pts[6, 0], pts[6, 1]), color_mapping[1], point_size)
        cv2.line(frame, (pts[7, 0], pts[7, 1]), (pts[6, 0], pts[6, 1]), color_mapping[1], point_size)

        # plot right front leg
        cv2.line(frame, (pts[3, 0], pts[3, 1]), (pts[8, 0], pts[8, 1]), color_mapping[2], point_size)
        cv2.line(frame, (pts[8, 0], pts[8, 1]), (pts[9, 0], pts[9, 1]), color_mapping[2], point_size)
        cv2.line(frame, (pts[10, 0], pts[10, 1]), (pts[9, 0], pts[9, 1]), color_mapping[2], point_size)

        # plot left back leg
        cv2.line(frame, (pts[4, 0], pts[4, 1]), (pts[11, 0], pts[11, 1]), color_mapping[6], point_size)
        cv2.line(frame, (pts[12, 0], pts[12, 1]), (pts[11, 0], pts[11, 1]), color_mapping[6], point_size)
        cv2.line(frame, (pts[12, 0], pts[12, 1]), (pts[13, 0], pts[13, 1]), color_mapping[6], point_size)

        # plot right back leg
        cv2.line(frame, (pts[4, 0], pts[4, 1]), (pts[14, 0], pts[14, 1]), color_mapping[7], point_size)
        cv2.line(frame, (pts[15, 0], pts[15, 1]), (pts[14, 0], pts[14, 1]), color_mapping[7], point_size)
        cv2.line(frame, (pts[15, 0], pts[15, 1]), (pts[16, 0], pts[16, 1]), color_mapping[7], point_size)
    return frame


def cv2_visualize_keypoints(frames, pts, savepath, idx, num_pts=17, order=0):
    inv_normalize = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                                        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])
    inputs = inv_normalize(frames)
    inputs = inputs.numpy().transpose(0, 2, 3, 1)
    for b in range(inputs.shape[0]):
        frame = np.uint8(inputs[b].copy() * 255)
        kpt = pts[b].astype(np.int)
        x = []
        y = []
        for i in range(num_pts):
            x.append(kpt[i, 0])
            y.append(kpt[i, 1])
            # plot keypoints on each image
            cv2.circle(frame, (x[-1], y[-1]), 2, (0, 255, 0), -1)
        frame = cv2_plot_lines(frame, kpt, order)
        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)
        cv2.imwrite(savepath + str(idx + b) + '.jpg', frame)