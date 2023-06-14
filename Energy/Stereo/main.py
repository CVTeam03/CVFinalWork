import cv2
import matplotlib.cm as cm
import os
import numpy as np
import disparity as disp
import time
from pfm import load_pfm

INPUT_DIR = 'input'
OUTPUT_DIR = 'myoutput'
source_img = {
    'adirondack': (61, 80, 0.25),
    'cones': (65, 87, None),
    'flowers': (150, 203, 0.25),
    'motorcycle': (60, 118, 0.25),
    'pipes': (70, 82, 0.25),
}


def get_accuracy(true, pred, scale):
    is_correct = np.abs(true - pred) <= 2. * scale
    total_accuracy = is_correct.sum() / float(true.size)
    is_visible = true >= 0
    visible_accuracy = is_correct[is_visible].sum() / float(is_visible.sum())
    return total_accuracy, visible_accuracy


def process_image_pair(image_pair, generate_ssd=False, generate_graphcut=False):
    search_depth, occlusion_cost, pfm_scale = source_img[image_pair]
    left = cv2.imread(os.path.join(INPUT_DIR, image_pair, 'im0.png'))
    right = cv2.imread(os.path.join(INPUT_DIR, image_pair, 'im1.png'))

    disparity_ssd, accuracy_ssd, disparity_graphcut, accuracy_graphcut, ground_truth = None, None, None, None, None

    if pfm_scale:
        ground_truth = load_pfm(os.path.join(INPUT_DIR, image_pair, 'disp0.pfm'), pfm_scale)

    if generate_ssd:
        disparity_ssd = disp.disparity(
            left, right,
            method=disp.SSD,
            search_depth=search_depth,
        )
        if pfm_scale is not None:
            accuracy_ssd = get_accuracy(ground_truth, disparity_ssd, pfm_scale)

    if generate_graphcut:
        disparity_graphcut = disp.disparity(
            left, right,
            method=disp.GRAPHCUT,
            search_depth=search_depth,
            occlusion_cost=occlusion_cost,
        )
        if pfm_scale is not None:
            accuracy_ssd = get_accuracy(ground_truth, disparity_graphcut, pfm_scale)

    return disparity_ssd, accuracy_ssd, disparity_graphcut, accuracy_graphcut


def disparity_to_gray(disp, bgr=True):
    image = np.zeros((disp.shape[0], disp.shape[1], 3), dtype=np.uint8)
    is_occluded = disp < 0
    image[:] = np.where(is_occluded, 0, 255 * disp / disp.max())[:, :, np.newaxis]
    image[is_occluded] = [255, 255, 0] if bgr else [0, 255, 255]
    return image


def disparity_to_jet(disp, bgr=True):
    cm_jet = cm.ScalarMappable(cmap='jet')
    is_occluded = disp < 0
    # print(is_occluded)
    jet = cm_jet.to_rgba(np.where(is_occluded, 0, disp), bytes=True)[:, :, :3]
    jet[is_occluded] = 0
    if not bgr:
        return jet
    return cv2.cvtColor(jet, cv2.COLOR_RGB2BGR)


class MyFinalSolver:
    def __init__(self, img):
        self.img = img

    def solve(self):
        for img_ in self.img:
            t3 = time.time()
            disparity_ssd_res = process_image_pair(img_, generate_ssd=True)[0]
            t4 = time.time()
            print("{} time ssd_cost is {}s".format(img_, t4 - t3))
            cv2.imwrite(os.path.join(OUTPUT_DIR, img_, 'ssd-gray.png'),
                        disparity_to_gray(disparity_ssd_res))
            t1 = time.time()
            disparity_gc_res = process_image_pair(img_, generate_graphcut=True)[0]
            t2 = time.time()
            print("{} time gc_cost is {}s".format(img_, t2 - t1))
            # print(os.path.join(OUTPUT_DIR, img_, 'ssd-gray.png'))
            cv2.imwrite(os.path.join(OUTPUT_DIR, img_, 'gc-gray.png'),
                        disparity_to_gray(disparity_gc_res))
            print("gc finish")


if __name__ == '__main__':
    solver = MyFinalSolver(source_img)
    solver.solve()
