import numpy as np
import random
import math


def add_gaussian_noise(img):
    mean = 0
    var = 0.01

    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (img.shape[0], img.shape[1], img.shape[2]))
    noise_img = img + gaussian

    return noise_img


def cutout(img):
    depth, height, width = img.shape
    assert(height == width)

    box_side = int(width/5) + int(random.SystemRandom().random() * (width/4 - width/5))
    box_depth = int(depth/5) + int(random.SystemRandom().random() * (depth/4 - depth/5))

    z = int(random.SystemRandom().random() * depth)
    y = int(random.SystemRandom().random() * height)
    x = int(random.SystemRandom().random() * width)

    z1 = np.clip(z - box_depth // 2, 0, depth)
    z2 = np.clip(z + box_depth // 2, 0, depth)
    y1 = np.clip(y - box_side // 2, 0, height)
    y2 = np.clip(y + box_side // 2, 0, height)
    x1 = np.clip(x - box_side // 2, 0, width)
    x2 = np.clip(x + box_side // 2, 0, width)

    img[z1:z2, y1:y2, x1:x2] = 0.

    return img


def generate_rotation_matrix(axis, angle):
    angle = np.radians(angle)
    axis = axis / np.linalg.norm(axis)

    a = math.cos(angle / 2.0)
    b, c, d = axis * math.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    return np.array([[aa + bb - cc - dd, 2 * (bc - ad), 2 * (bd + ac), 0],
                     [2 * (bc + ad), aa + cc - bb - dd, 2 * (cd - ab), 0],
                     [2 * (bd - ac), 2 * (cd + ab), aa + dd - bb - cc, 0],
                     [0, 0, 0, 1]])


def generate_random_rotation_matrix(axis, angle_range):
    angle = random.SystemRandom().random() * 2 * angle_range - angle_range

    return generate_rotation_matrix(axis, angle)
