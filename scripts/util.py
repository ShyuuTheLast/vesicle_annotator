import os
import cv2


def mkdir(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def write_txt(filename, content):
    with open(filename, "w") as a:
        if isinstance(content, (list,)):
            for ll in content:
                a.write(ll)
                if "\n" not in ll:
                    a.write("\n")
        else:
            a.write(content)


def read_txt(filename):
    with open(filename) as a:
        content = a.readlines()
    return content


def rotate_image(image, angle, center=None, scale=1.0):
    if angle == 0:
        return image
    # grab the dimensions of the image
    (h, w) = image.shape[:2]
    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)
    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, M, (w, h))
