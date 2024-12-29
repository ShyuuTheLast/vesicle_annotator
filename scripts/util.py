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
