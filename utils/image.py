import cv2
import torch
import numpy as np
from numpy import random


def pre_process(img, resolution, mean, std):
    height, width = img.shape[:2]
    input_h = input_w = resolution
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0
    trans_input = get_affine_transform(c, s, [input_h, input_w])
    inv_trans = get_affine_transform(c, s, [input_h, input_w], 1)
    inv_trans = inv_trans.astype(np.float32)
    input_img = cv2.warpAffine(
        img, trans_input, (input_w, input_h))
    input_img = ((input_img / 255. - mean) / std).astype(np.float32)
    image = input_img.transpose(2, 0, 1)
    image = torch.from_numpy(image)
    return image, inv_trans


def get_affine_transform(center, scale, output_size, inv=0):
    dst_w = output_size[0]
    dst_h = output_size[1]
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src_dir = get_dir([0, scale * -0.5])
    dst_dir = np.array([0, dst_w * -0.5], np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    # print(trans.dtype)
    trans = trans.astype(np.float32)
    return trans


def get_dir(src_point):
    src_result = [0, 0]
    src_result[0] = src_point[0]
    src_result[1] = src_point[1]

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img


def affine_transform(trans, box):
    h = box.shape[0]
    box1 = np.concatenate((box[:, 0:2], np.ones(h, 1)), 1).T
    box2 = np.concatenate((box[:, 2:4], np.ones(h, 1)), 1).T
    box1 = np.dot(trans, box1).T
    box2 = np.dot(trans, box2).T
    return np.concatenate((box1, box2), 1)


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3,))
    image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)


def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)


def affine_transform_torch(trans, box):
    box = box.view(-1, 4)
    h = box.shape[0]
    box1 = torch.cat((box[:, 0:2], torch.ones(h, 1)), 1)
    box2 = torch.cat((box[:, 2:4], torch.ones(h, 1)), 1)
    trans = trans.transpose(0, 1).contiguous()
    box1 = torch.matmul(box1, trans)
    box2 = torch.matmul(box2, trans)
    return torch.cat((box1, box2), 1).view(-1, 4)
