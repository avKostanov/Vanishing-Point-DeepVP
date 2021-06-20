
import os
import random
import numpy as np
import json
import cv2
import config
from math import degrees

# Dataset Generation



def read_filenames(path):
    return os.listdir(os.path.join(path, 'source/'))


def read_markup(path):
    f = open(os.path.join(path, "markup.json"))
    return json.load(f)


def select_test_imgs(fnames, n, seed):
    random.seed(seed)
    return random.sample(fnames, n)


def generate_random_angles(num, seed):
    np.random.seed(seed)
    return ((np.random.random_sample(num) - 0.5) / 150, (np.random.random_sample(num) - 0.5) / 150)


def generate_random_shifts(num, seed):
    np.random.seed(seed)
    return ((np.random.random_sample(num) - 0.5) * 20, (np.random.random_sample(num) - 0.5) * 20)


def generate_matrix(phi, theta, dx, dy, img_size):
    height, width, _ = img_size

    A1 = np.array([[1,             0,              -width/2],
                   [0,             1,              -height/2],
                   [0,             0,              1],
                   [0,             0,              1]])

    # Rotation matrices around the X and Y axis
    Rx = np.array([[1, 0,              0,              0],
                   [0, np.cos(theta),  -np.sin(theta), 0],
                   [0, np.sin(theta),  np.cos(theta), 0],
                   [0, 0,              0,              1]])

    Ry = np.array([[np.cos(phi),  0,   -np.sin(phi),  0],
                   [0,            1,   0,             0],
                   [np.sin(phi),  0,   np.cos(phi),   0],
                   [0,            0,   0,             1]])

    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(Rx, Ry)

    # Projection 3D -> 2D matrix
    A2 = np.array([[1,    0,   width/2 + dx,   0],
                   [0,    1,   height/2 + dy,  0],
                   [0,    0,   1,         0]])

    # Final transformation matrix
    return np.dot(A2, np.dot(R, A1))


def calculate_upper_left(ul, bl, ur):
    y = ul[1] if ul[1] > ur[1] else ur[1]
    y = 0 if y < 0 else y

    x = ul[0] if ul[0] > bl[0] else bl[0]
    x = 0 if x < 0 else x

    return (int(y), int(x))


def calculate_bottom_right(ur, br, bl):
    y = bl[1] if bl[1] < br[1] else br[1]
    y = 300 if y > 300 else y

    x = ur[0] if ur[0] < br[0] else br[0]
    x = 300 if x > 300 else x

    return (int(y), int(x))


def calculate_roi(R, img_size):
    height, width, _ = img_size

    # NOTE: here points is described in (x,y) order
    vertices = {}
    vertices['upper_left'] = [0, 0]
    vertices['upper_right'] = [width - 1, 0]
    vertices['bottom_left'] = [0, height - 1]
    vertices['bottom_right'] = [width - 1, height - 1]

    rotated_vertices = {}
    for key, val in vertices.items():
        val.append(1)
        tmp = np.dot(R, np.array(val, dtype=float))
        tmp /= tmp[2]
        rotated_vertices[key] = list(tmp[0:2])

    # NOTE: points are returned in (y,x) order
    p1 = calculate_upper_left(rotated_vertices['upper_left'],
                              rotated_vertices['bottom_left'],
                              rotated_vertices['upper_right'])

    p2 = calculate_bottom_right(rotated_vertices['upper_right'],
                                rotated_vertices['bottom_right'],
                                rotated_vertices['bottom_left'])

    return p1, p2


def normalize_img(img, roi):
    p1, p2 = roi

    height, width, _ = img.shape
    scale = ((height - 1) / (p2[0] - p1[0]), (width - 1) / (p2[1] - p1[1]))

    crop_img = img[p1[0]:p2[0], p1[1]:p2[1]]
    resized_img = cv2.resize(crop_img, (width, height))

    return resized_img, scale


def transform_img(img, R):
    rotated_img = cv2.warpPerspective(
        img.copy(), R, (img.shape[0], img.shape[1]))
    roi = calculate_roi(R, img.shape)
    resized_img, scale = normalize_img(rotated_img, roi)

    return resized_img, scale, roi


def transform_answer(ans, R, scale, roi):
    # NOTE: ans in (x, y) format
    p1, _ = roi

    scaley, scalex = scale
    ans.append(1)
    ans = np.array(ans)

    rotated_ans = np.dot(R, ans)
    rotated_ans /= rotated_ans[2]

    if p1[1] > 0:
        rotated_ans[0] -= p1[1]

    if p1[0] > 0:
        rotated_ans[1] -= p1[0]

    new_ans = rotated_ans[0:2]
    new_ans[0] *= scalex
    new_ans[1] *= scaley

    return list(new_ans)


def save_image(path, img):
    cv2.imwrite(path, img)


def save_markup(path, markup):
    f = open(os.path.join(path, "markup.json"), "w")
    json.dump(markup, f)


def generate_test(dst_path, markup, fnames, src_path, seed):
    num = len(fnames)
    phi, theta = generate_random_angles(num, seed)
    dx, dy = generate_random_shifts(num, seed)

    list_to_iterate = zip(fnames, phi, theta, dx, dy)
    new_markup = {}
    for file, phi, theta, dx, dy in list_to_iterate:
        img = cv2.imread(os.path.join(src_path, 'source/', file))
        answer = markup[file]  # currently wrong
        T = generate_matrix(phi, theta, dx, dy, img.shape)
        transformed_img, scale, roi = transform_img(img, T)
        transformed_answer = transform_answer(answer, T, scale, roi)
        new_markup[file] = transformed_answer
        save_image(os.path.join(dst_path, file), transformed_img)

    save_markup(dst_path, new_markup)


def generate_ds(dst_path, src_path, num, seed):
    fnames = read_filenames(src_path)
    markup = read_markup(src_path)
    selected_fnames = select_test_imgs(fnames, num, seed)
    generate_test(dst_path, markup, selected_fnames, src_path, seed)


def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def get_shapes(path):
    names = os.listdir(path)
    try:
        names.remove('markup.json')
    except ValueError:
        pass

    return {name: cv2.imread(os.path.join(path, name)).shape[0:2] for name in names if not name.endswith('.csv')}

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def calculate_angle(ans, gt, shape):
    n,m = shape
    
    O = np.array((n/2, m/2, ((n/2)**2 + (m/2)**2)**0.5))
    A = np.array((ans[0], ans[1], 0))
    GT = np.array((gt[0], gt[1], 0))

    return angle_between(A - O, GT - O)


def calc_metrics(answers_dict, gt_answers, path_to_imgs):
    answers = answers_dict
    gt = gt_answers
    shapes = get_shapes(path_to_imgs)


    angles = []
    for name in gt:
        angle = calculate_angle(answers[name], gt[name], shapes[name])
        angles.append(angle)

    angles = np.array(angles)

    return np.mean(angles), np.median(angles)