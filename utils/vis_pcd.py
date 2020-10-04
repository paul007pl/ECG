import h5py
import open3d as o3d
import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import io
import os


def get_pts(pcd):
    points = np.asarray(pcd.points)
    X = []
    Y = []
    Z = []
    for pt in range(points.shape[0]):
        X.append(points[pt][0])
        Y.append(points[pt][1])
        Z.append(points[pt][2])

    return np.asarray(X), np.asarray(Y), np.asarray(Z)


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_pcd(pcd_list, pdf, object_id, view_id):
    num_pcds = len(pcd_list)
    fig = plt.figure(figsize=(60, 60))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    for ind, pcd_original in enumerate(pcd_list):
        pcd = o3d.geometry.PointCloud(pcd_original)
        translation_matrix = np.asarray(
            [[1, 0, 0, ind - int(num_pcds / 2 - 0.5)],
             [0, 1, 0, 0.275 * ind - (num_pcds / 2 - 0.5)],
             [0, 0, 1, 0],
             [0, 0, 0, 1]])
        rotation_matrix = np.asarray(
            [[1, 0, 0, 0],
             [0, 0, -1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]])
        transform_matrix = rotation_matrix @ translation_matrix
        pcd = pcd.transform(transform_matrix)
        X, Y, Z = get_pts(pcd)
        t = Z
        ax.scatter(X, Y, Z, c=t, cmap='jet', marker='o', s=1)
        ax.grid(False)
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    set_axes_equal(ax)
    if view_id != 0:
        plt.title("   Obj: %d Scan: %d" % (object_id, view_id), y=0.62, fontsize=20)
    else:
        gt_title = "    GT"
        input_title = "                                     Input"
        pcn_title = "                                pcn"
        topnet_title = "                                TopNet"
        msn_title = "                                MSN"
        cascade_title = "                                Cascaded"
        title = gt_title + input_title + pcn_title + topnet_title + msn_title + cascade_title
        plt.title(title + "\n   Obj: %d Scan: %d" % (object_id, view_id), y=0.62, fontsize=20)

    plt.axis('off')
    bbox = fig.bbox_inches.from_bounds(19, 27, 25, 7)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches=bbox)
    buf.seek(0)
    im = Image.open(buf)
    plt.close()
    return im


if __name__ == '__main__':
    model_type = sys.argv[1]  # cd or emd
    best_type = sys.argv[2]  # cd_t or cd_p or emd
    results_prefix = sys.argv[3]

    # outputs by 4 models (pcn, Topnet, MSN, Cascade)
    output_file_pcn = h5py.File(os.path.join(results_prefix, 'pcn_%s/best_%s_network_pcds.h5' % (model_type, best_type)), 'r')
    pcn = output_file_pcn['output_pcds'][()]
    output_file_topnet = h5py.File(os.path.join(results_prefix,'topnet_%s/best_%s_network_pcds.h5' % (model_type, best_type)), 'r')
    topnet = output_file_topnet['output_pcds'][()]
    output_file_msn = h5py.File(os.path.join(results_prefix,'msn_%s/best_%s_network_pcds.h5' % (model_type, best_type)), 'r')
    msn = output_file_msn['output_pcds'][()]
    output_file_cascade = h5py.File(os.path.join(results_prefix,'cascade_%s/best_%s_network_pcds.h5' % (model_type, best_type)), 'r')
    cascade = output_file_cascade['output_pcds'][()]
    # gt
    gt_file = h5py.File('/mnt/lustre/chenxinyi1/pl/data_generation/my_dataset1/my_test_gt_data_2048_1.h5', 'r')
    novel_gt = gt_file['novel_complete_pcds'][()]
    gt = gt_file['complete_pcds'][()]
    gt = np.concatenate((gt, novel_gt), axis=0)
    # input
    input_file = h5py.File('/mnt/lustre/chenxinyi1/pl/data_generation/my_dataset1/my_test_input_data_denoised_1.h5', 'r')
    novel_input = input_file['novel_incomplete_pcds'][()]
    inputs = input_file['incomplete_pcds'][()]
    inputs = np.concatenate((inputs, novel_input), axis=0)

    to_plot = [7, 10, 15, 17, 55, 95, 123, 132, 133, 136,
               183, 191, 192, 243, 249, 253, 254, 261, 266, 269,
               303, 311, 323, 357, 367, 400, 405, 419, 434, 449,
               459, 483, 561, 596,
               601, 605, 612, 614, 630, 638, 646, 652, 668, 673,
               787, 793, 799, 807, 823, 829, 851, 864, 879, 895,
               902, 911, 913, 926, 930, 935, 960, 993, 1007, 1011,
               1055, 1063, 1069, 1072, 1075, 1076, 1082, 1104, 1108, 1130,
               1204, 1227, 1248,
               1258, 1271, 1276, 1277, 1282,
               1329, 1349,
               1364, 1377, 1382, 1388,
               1414, 1416, 1420,
               1452, 1454, 1456, 1478, 1479,
               1500, 1508, 1510, 1514, 1515, 1521, 1529, 1549,
               1565, 1574]
    page_list = []
    for ind, i in enumerate(to_plot):
        print('%d/%d' % (ind, len(to_plot)))
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(gt[i])
        width, height = 2500, 700
        concat_im = Image.new('RGB', (width, 26 * height))
        for j in range(0, 26):
            input_pcd = o3d.geometry.PointCloud()
            input_pcd.points = o3d.utility.Vector3dVector(inputs[i * 26 + j])
            pcn_pcd = o3d.geometry.PointCloud()
            pcn_pcd.points = o3d.utility.Vector3dVector(pcn[i * 26 + j])
            topnet_pcd = o3d.geometry.PointCloud()
            topnet_pcd.points = o3d.utility.Vector3dVector(topnet[i * 26 + j])
            msn_pcd = o3d.geometry.PointCloud()
            msn_pcd.points = o3d.utility.Vector3dVector(msn[i * 26 + j])
            cascade_pcd = o3d.geometry.PointCloud()
            cascade_pcd.points = o3d.utility.Vector3dVector(cascade[i * 26 + j])
            pcds = plot_pcd([gt_pcd, input_pcd, pcn_pcd, topnet_pcd, msn_pcd, cascade_pcd], i, j)
            concat_im.paste(pcds, (0, j * height))
        page_list.append(concat_im)
    first = page_list.pop(0)
    first.save(os.path.join(results_prefix, '%s_train_best_%s.pdf' % (model_type, best_type)),
               "PDF", resolution=100.0, optimize=True, save_all=True, append_images=page_list)






