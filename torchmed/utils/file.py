from shutil import copyfile
import os
import numpy as np
import cv2
import matplotlib.pyplot


def copy_file(source, destination=None, append='', prepend=''):
    """
    Copy a file to a destination folder/file.
    If the destination is not specified, then the file is copied in the same
    directory as the source.
    """
    if destination is None:
        destination = os.path.dirname(source)
    filename = os.path.basename(source)
    file_wo_ext = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1]

    # if the destination is a directory we can append prepend on the filename
    if os.path.isdir(destination):
        copyfile(source, os.path.join(destination,
                                      append + file_wo_ext + prepend + ext))
    else:
        copyfile(source, destination)


def export_to_file(tensor, file):
    """
    Export a torch tensor to file
    """
    from scipy import misc
    misc.imsave(file, tensor.numpy())


def write_image_segmentation(img, target, nb_classes, output_dir):
    def get_annotated_image(tensor, n_labels, colors):
        temp = tensor.numpy()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()

        for l in range(0, n_labels):
            r[temp == l] = colors[l, 0]
            g[temp == l] = colors[l, 1]
            b[temp == l] = colors[l, 2]

        # for unwanted labels
        r[temp == -1] = 255
        g[temp == -1] = 255
        b[temp == -1] = 255

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b

        return rgb

    def get_spaced_colors(n):
        max_value = 16581375  # 255**3
        interval = int(max_value / n)
        colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
        return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    seg_colors = np.array(get_spaced_colors(nb_classes))
    for n in range(0, img.size(0)):
        brain = cv2.cvtColor(img[n].numpy(), cv2.COLOR_GRAY2BGR)
        brain = cv2.normalize(brain, brain, 0, 255, cv2.NORM_MINMAX)
        seg_brain = get_annotated_image(target[n], nb_classes, seg_colors)
        res = np.concatenate([brain, seg_brain], axis=1).astype(int)

        output_file = os.path.join(output_dir, str(n) + '.png')
        matplotlib.pyplot.imsave(output_file, res)
