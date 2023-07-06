import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import os
import argparse


def gen_rand_dots(num, var_max, r, bound_x, bound_y):
    X, Y = np.mgrid[-1 * bound_x:bound_x:complex(0, num), -1 * bound_y:bound_y:complex(0, num)]
    noise_x = np.random.uniform(low=-1 * var_max, high=var_max, size=(num * num))
    noise_y = np.random.uniform(low=-1 * var_max, high=var_max, size=(num * num))
    global_noise_x = np.random.uniform(low=-0.03, high=0.03, size=())
    global_noise_y = np.random.uniform(low=-0.03, high=0.03, size=())
    X = np.reshape(X, (-1)) + noise_x + global_noise_x
    Y = np.reshape(Y, (-1)) + noise_y + global_noise_y
    Z = np.zeros_like(X)
    centers = np.stack([X, Y, Z], axis=-1)

    rs = np.ones_like(X) * r

    return np.array(centers), np.array(rs)


def render_concave(bound_flat_x, rez, centers, rs, fov, slant, base_color=None, dot_color=None):
    '''
    Render the concave surface.

    :param bound_flat_x: the boundary along x-axis on the flat surface;
    :param rez: resolution;
    :param centers: the centers of the dots;
    :param rs: the radii of the dots;
    :param base_color: base color;
    :param dot_color: dot color;
    :param fov: the field of view of the camera;
    :param slant: the physical slant of the dihedral face;
    :return: a rendered image;
    '''

    fov = fov / 180. * np.pi
    slant = slant / 180. * np.pi

    if dot_color is None:
        dot_color = np.array([0., 0., 0.])
    if base_color is None:
        base_color = np.array([255., 255., 255.])

    bound = bound_flat_x * np.cos(slant)
    X, Y = np.mgrid[-1:1:complex(0, rez), -1:1:complex(0, rez)]
    X = X * bound
    Y = Y * bound

    # calculate the camera position; the image plane should precisely cover the entire width of the dihedral face
    eye = np.array([0., 0., bound_flat_x * np.sin(slant) + bound_flat_x * np.cos(slant) / np.tan(fov / 2.)])

    plane = np.vstack((np.vstack((X.flatten(), Y.flatten())), np.sin(slant) * np.ones_like(X.flatten()))).T
    directions = (plane - eye) / np.linalg.norm(plane - eye, axis=-1, keepdims=True)
    origin = np.tile(eye, (len(directions), 1))

    # calculate the intersection point of rays to the surface
    # if x is positive
    t_positive = (np.tan(slant) * origin[:, 0] - origin[:, 2]) / (directions[:, 2] - np.tan(slant) * directions[:, 0])
    p1 = origin + t_positive[:, None] * directions
    p1 = p1 * (t_positive > 0)[:, None]  # if t is negative, set point to origin
    # if x is negative
    t_negative = (-1 * np.tan(slant) * origin[:, 0] - origin[:, 2]) / (
            directions[:, 2] + np.tan(slant) * directions[:, 0])
    p2 = origin + t_negative[:, None] * directions
    p2 = p2 * (t_negative > 0)[:, None]  # if t is negative, set point to origin
    # the actual intersections
    points = p1 * (p1[:, 0] > 0)[:, None] + p2 * (p2[:, 0] < 0)[:, None]

    # maps the points on the dihedral surface to the flat surface
    flat_x = points[:, 0] / np.cos(slant)
    points_flat = np.stack([flat_x, points[:, 1], np.zeros_like(flat_x)], axis=-1)

    # checks whether an intersection point fails into one of the dots
    intersection_map = []
    for center, r in zip(centers, rs):
        d = np.linalg.norm(center - points_flat, axis=-1)
        is_intersect = d < r
        intersection_map.append(is_intersect)
    intersection_map = np.stack(intersection_map, 0)
    intersection_map = np.any(intersection_map, axis=0)

    rgb = base_color * (1 - intersection_map[:, None]) + dot_color * intersection_map[..., None]
    rgb = np.reshape(rgb, (rez, rez, 3))
    rgb = np.transpose(rgb, (1, 0, 2))

    return rgb


def render_convex(bound_flat_x, rez, centers, rs, fov, slant, base_color=None, dot_color=None):
    '''
    Render the convex surface.

    :param bound_flat_x: the boundary along x-axis on the flat surface;
    :param rez: resolution;
    :param centers: the centers of the dots;
    :param rs: the radii of the dots;
    :param base_color: base color;
    :param dot_color: dot color;
    :param fov: the field of view of the camera;
    :param slant: the physical slant of the dihedral face;
    :return: a rendered image;
    '''

    fov = fov / 180. * np.pi
    slant = slant / 180. * np.pi

    if dot_color is None:
        dot_color = np.array([0., 0., 0.])
    if base_color is None:
        base_color = np.array([255., 255., 255.])

    bound = bound_flat_x * np.cos(slant)
    X, Y = np.mgrid[-1:1:complex(0, rez), -1:1:complex(0, rez)]
    X = X * bound
    Y = Y * bound

    # calculate the camera position; the image plane should precisely cover the entire width of the dihedral face
    eye = np.array([0., 0., bound_flat_x * np.cos(slant) / np.tan(fov / 2.)])

    plane = np.vstack((np.vstack((X.flatten(), Y.flatten())), np.zeros_like(X.flatten()))).T
    directions = (plane - eye) / np.linalg.norm(plane - eye, axis=-1, keepdims=True)
    origin = np.tile(eye, (len(directions), 1))

    # calculate the intersection point of rays to the surface
    # if x is positive
    t_positive = (bound * np.tan(slant) - np.tan(slant) * origin[:, 0] - origin[:, 2]) \
                 / (directions[:, 2] + np.tan(slant) * directions[:, 0])
    p1 = origin + t_positive[:, None] * directions
    p1 = p1 * (t_positive > 0)[:, None]  # if t is negative, set point to origin
    # if x is negative
    t_negative = (bound * np.tan(slant) + np.tan(slant) * origin[:, 0] - origin[:, 2]) \
                 / (directions[:, 2] - np.tan(slant) * directions[:, 0])
    p2 = origin + t_negative[:, None] * directions
    p2 = p2 * (t_negative > 0)[:, None]  # if t is negative, set point to origin

    # the actual intersections
    points = p1 * (p1[:, 0] > 0)[:, None] + p2 * (p2[:, 0] < 0)[:, None]

    # maps the points on the dihedral surface to the flat surface
    flat_x = points[:, 0] / np.cos(slant)
    points_flat = np.stack([flat_x, points[:, 1], np.zeros_like(flat_x)], axis=-1)

    # checks whether an intersection point fails into one of the dots
    intersection_map = []
    for center, r in zip(centers, rs):
        d = np.linalg.norm(center - points_flat, axis=-1)
        is_intersect = d < r
        intersection_map.append(is_intersect)
    intersection_map = np.stack(intersection_map, 0)
    intersection_map = np.any(intersection_map, axis=0)

    rgb = base_color * (1 - intersection_map[:, None]) + dot_color * intersection_map[..., None]
    rgb = np.reshape(rgb, (rez, rez, 3))
    rgb = np.transpose(rgb, (1, 0, 2))

    return rgb


# render images
def render_data(data_dir, fovs, optical_slants, loc_dis, grid_len, dot_size, bound, repeat):
    reps = range(repeat)
    for rep in reps:
        print("repetition = " + str(rep))
        for dis in loc_dis:
            for fov in fovs:
                print("fov = " + str(fov))
                for optical_slant in optical_slants:
                    print("optical slant = " + str(optical_slant))
                    # generate dot patterns
                    centers, rs_deformed = gen_rand_dots(num=grid_len, r=dot_size, var_max=dis, bound_x=bound, bound_y=bound)

                    physical_slant_concave = optical_slant + fov / 4
                    physical_slant_convex = optical_slant - fov / 4
                    img_concave = render_concave(bound_flat_x=bound, rez=rez, centers=centers, rs=rs_deformed,
                                                 fov=fov, slant=physical_slant_concave)
                    img_convex = render_convex(bound_flat_x=bound, rez=rez, centers=centers, rs=rs_deformed,
                                               fov=fov, slant=physical_slant_convex)
                    plt.imsave(
                        os.path.join(data_dir, 'concave_rep_%02d_fov_%.2f_opt_slant_%.3f_phy_slant_%.3f_var_loc_%d.png'
                                     % (rep, fov, optical_slant, physical_slant_concave, int(dis / loc_dis[1]))),
                        img_concave / 255.)
                    plt.imsave(
                        os.path.join(data_dir, 'convex_rep_%02d_fov_%.2f_opt_slant_%.3f_phy_slant_%.3f_var_loc_%d.png'
                                     % (rep, fov, optical_slant, physical_slant_convex, int(dis / loc_dis[1]))),
                        img_convex / 255.)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="data_exp2", help='Name of the dataset')
    parser.add_argument('--rez', type=int, default=256)
    parser.add_argument('--grid_len', type=int, default=12, help='Number of dots per row/column')
    parser.add_argument('--nb_fovs', type=int, default=12, help='Number of the FOV values')
    parser.add_argument('--nb_slants', type=int, default=8, help='Number of the optical slant values')
    parser.add_argument('--irr_level', type=int, default=5, help='Number of irregularity levels')
    parser.add_argument('--repeat_train', type=int, default=10)
    parser.add_argument('--repeat_test', type=int, default=2)
    parser.add_argument('--batched_job', type=bool, default=False)
    args = parser.parse_args()

    dataset = args.dataset
    nb_fovs = args.nb_fovs
    rez = args.rez
    grid_len = args.grid_len
    nb_slants = args.nb_slants
    irr_level = args.irr_level
    repeat_train = args.repeat_train
    repeat_test = args.repeat_test
    batched = args.batched_job

    out_dir_train = os.path.join('./datasets', dataset, 'train')
    out_dir_test = os.path.join('./datasets', dataset, 'test')
    os.makedirs(out_dir_train, exist_ok=True)
    os.makedirs(out_dir_test, exist_ok=True)

    bound = 1.
    dot_size = 0.05
    # light_angle = 15.
    fovs = np.linspace(5, 60, nb_fovs)
    optical_slants = np.linspace(25, 60, nb_slants)
    loc_dis = np.linspace(0, 0.08, irr_level)  # dot location displacement upper bound

    render_data(out_dir_train, fovs, optical_slants, loc_dis, grid_len, dot_size, bound, repeat_train)
    render_data(out_dir_test, fovs, optical_slants, loc_dis, grid_len, dot_size, bound, repeat_test)

