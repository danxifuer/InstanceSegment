import numpy as np
import cv2
import glob
import os
import random
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from multiprocessing import Pool


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def draw_grid(im, grid_size):
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))
    return im


class CreateImage:
    def __init__(self, img_root_lst, save_img_root,
                 save_inst_root, make_argument, flag):
        self.make_argument = make_argument
        self.save_img_root = save_img_root
        self.save_inst_root = save_inst_root
        self.img_root_lst = img_root_lst
        self.flag = str(flag)
        self.alpha = 2
        self.sigma = 0.11
        self.alpha_affine = 0.07
        if not os.path.exists(save_img_root):
            os.makedirs(save_img_root)
        if not os.path.exists(save_inst_root):
            os.makedirs(save_inst_root)

    @staticmethod
    def _get_images(root):
        return glob.glob(os.path.join(root, '*.png')) + glob.glob(os.path.join(root, '*.jpg'))

    @staticmethod
    def _merge_masks(mask_list, height, width):
        new_mask = np.zeros(shape=(height, width), dtype=np.int32)
        for k in range(len(mask_list)):
            sub_mask = cv2.imread(mask_list[k], 0)
            assert sub_mask.shape == new_mask.shape, '%s vs %s' % (sub_mask.shape, new_mask.shape)
            sub_mask = sub_mask.astype(np.int32)
            sub_mask[sub_mask > 0] = k + 1
            new_mask += sub_mask
            new_mask[new_mask > k + 1] = k + 1
        channel_1 = new_mask.copy()
        channel_2 = new_mask.copy()
        channel_1[channel_1 > 255] = 0
        channel_2[channel_2 <= 255] = 255
        channel_2 = channel_2 - 255
        assert np.max(channel_1) <= 255
        assert np.max(channel_2) <= 255
        return channel_1, channel_2

    @staticmethod
    def _rotate(image):
        height, width, = image.shape[:2]
        angle = random.choice([0, 90, 180, 270])
        # cv2.imshow('', image)
        # cv2.waitKey()
        rotate_mat = cv2.getRotationMatrix2D((width / 2, height / 2),
                                             angle, 1.0)

        rotated = cv2.warpAffine(image, rotate_mat, (width, height))
        # cv2.imshow('', rotated)
        # cv2.waitKey()
        return rotated

    def _create(self, single_img_root):
        # suffix_root = single_img_root.split('/')[-1]
        img_path = os.path.join(single_img_root, 'images')
        mask_path = os.path.join(single_img_root, 'masks')
        ori_image = self._get_images(img_path)
        assert len(ori_image) == 1
        ori_image = ori_image[0]
        mask_image_names = self._get_images(mask_path)
        cv_image = cv2.imread(ori_image, 0)
        height, width = cv_image.shape
        channel_1, channel_2 = self._merge_masks(mask_image_names, height, width)
        if self.make_argument:
            channel_1 = channel_1.astype(np.uint8)
            channel_2 = channel_2.astype(np.uint8)
            if len(mask_image_names) > 5:
                print(len(mask_image_names))
                # cv2.imshow('', channel_1 * int((255 / np.max(channel_1))))
                # cv2.waitKey()
                # cv2.imshow('', channel_2 * int((255 / (np.max(channel_2) + 1))))
                # cv2.waitKey()
            tri_channel_img = np.concatenate((cv_image[..., None],
                                              channel_1[..., None],
                                              channel_2[..., None]),
                                             axis=2)
            tri_channel_img = self._rotate(tri_channel_img)
            tri_channel_img = elastic_transform(tri_channel_img, width * self.alpha,
                                                width * self.sigma,
                                                width * self.alpha_affine)

            cv_image = tri_channel_img[..., 0].astype(np.uint8)
            # cv2.imshow('', cv_image * int((255 / np.max(cv_image))))
            # cv2.waitKey()
            channel_1 = tri_channel_img[..., 1]
            channel_2 = tri_channel_img[..., 2]
        channel_1 = channel_1.astype(np.int32)
        channel_2 = channel_2.astype(np.int32)
        # print(np.max(channel_2))
        channel_2[channel_2 > 0] = channel_2[channel_2 > 0] + 255
        mask_merged = channel_1 + channel_2

        # normalize image
        max_val = np.max(cv_image)
        ratio = 255 / max_val
        cv_image = cv_image * ratio

        bincount = np.bincount(mask_merged)
        if bincount.shape[0] <= 1 or np.sum(bincount == 0) > 0:
            save_image_path = os.path.join(self.save_img_root)
            save_mask_path = os.path.join(self.save_inst_root)
            if not os.path.exists(save_image_path):
                os.makedirs(save_image_path)
            if not os.path.exists(save_mask_path):
                os.makedirs(save_image_path)
            ori_image_name = os.path.basename(ori_image)
            cv2.imwrite("%s/%s" % (save_image_path, self.flag + '_' + ori_image_name), cv_image)
            np.save("%s/%s.npy" % (save_mask_path, self.flag + '_' + os.path.splitext(ori_image_name)[0]), mask_merged)

    def create(self):
        sub_img_root_list = open(self.img_root_lst, 'r').readlines()
        sub_img_root_list = list(map(lambda x: x.strip(), sub_img_root_list))
        pool = Pool(8)
        pool.map(creator._create, sub_img_root_list)


if __name__ == '__main__':
    lst = '/home/daiab/machine_disk/work/kaggle_nuclei/data/stage1_train.lst'
    save_img_root = '/home/daiab/machine_disk/work/kaggle_nuclei/data/LikeVOC/img'
    save_inst_root = '/home/daiab/machine_disk/work/kaggle_nuclei/data/LikeVOC/inst'
    creator = CreateImage(lst, save_img_root, save_inst_root, False, 0)
    creator.create()
    # for i in range(1, 6):
    #     creator = CreateImage(lst, save_img_root, save_inst_root, True, i)
    #     creator.create()
