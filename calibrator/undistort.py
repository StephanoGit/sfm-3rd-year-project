import argparse
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom


def get_text(parent, tag):
    element = parent.find(tag)
    return float(element.text) if element is not None else None


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Undistort image using camera intrinsics (.xml file only)"
    )

    parser.add_argument("camera", help="<input/file/path> | .xml")
    parser.add_argument("image", help="<image/file/path>")
    parser.add_argument("output", help="<output/file/path>")

    return parser.parse_args()


args = arg_parser()
print(args)

tree = ET.parse(args.camera)
root = tree.getroot()


# Access the <K> and <D> elements and check if they are not None
K = root.find("K")
if K is not None:
    fx = get_text(K, "fx")
    fy = get_text(K, "fy")
    cx = get_text(K, "cx")
    cy = get_text(K, "cy")

    K_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    print("K matrix:\n", K_matrix)
else:
    print("The element <K> was not found in the XML file.")

D = root.find("D")
if D is not None:
    k1 = get_text(D, "k1")
    k2 = get_text(D, "k2")
    p1 = get_text(D, "p1")
    p2 = get_text(D, "p2")
    k3 = get_text(D, "k3")

    D_matrix = np.array([k1, k2, p1, p2, k3])
    print("Distortion coefficients:\n", D_matrix)
else:
    print("The element <D> was not found in the XML file.")

img = cv2.imread(args.image)
if img is None:
    print("Failed to load image")
    exit(1)

K_undist = K_matrix
img_undist = cv2.undistort(img, K_matrix, D_matrix, newCameraMatrix=K_undist)
print("new K matrix:\n", K_undist)

name, ext = os.path.splitext(os.path.basename(args.image))
cv2.imwrite(os.path.join(args.output, name + "_undist" + ext), img_undist)
print("DONE")
