import argparse
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom


def export_intrinsics_to_xml(K, d, file_name):
    root = ET.Element("Intrinsics")

    K_node = ET.SubElement(root, "K")
    ET.SubElement(K_node, "fx").text = str(K[0][0])
    ET.SubElement(K_node, "fy").text = str(K[1][1])
    ET.SubElement(K_node, "cx").text = str(K[0][2])
    ET.SubElement(K_node, "cy").text = str(K[1][2])

    D_node = ET.SubElement(root, "D")
    ET.SubElement(D_node, "k1").text = str(d[0])
    ET.SubElement(D_node, "k2").text = str(d[1])
    ET.SubElement(D_node, "p1").text = str(d[2])
    ET.SubElement(D_node, "p2").text = str(d[3])
    ET.SubElement(D_node, "k3").text = str(d[4])

    xmlstr = minidom.parseString(ET.tostring(root, "utf-8")).toprettyxml(indent="   ")

    with open(file_name, "w") as f:
        f.write(xmlstr)


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Calibrate camera using a video of a checkerboard pattern"
    )

    parser.add_argument("input", help="<input/file/path>")
    parser.add_argument("output", help="<output/file/path> | .xml")

    parser.add_argument(
        "--debug-dir", help="<debug/dir/file/path (default: None)>", default=None
    )
    parser.add_argument(
        "--frame-step", help="use every N-th frame (default: 20)", default=20, type=int
    )
    parser.add_argument(
        "--max-frames",
        help="frame limit (default: None)",
        default=None,
        type=int,
    )

    return parser.parse_args()


args = arg_parser()
print(args)

source = cv2.VideoCapture(args.input)

pattern_size = (9, 6)
pattern_pts = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_pts[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

obj_pts, img_pts = [], []
height, width = 0, 0
frame = -1
used_frames = 0

while True:
    frame += 1

    retval, img = source.read()
    if not retval:
        break

    if frame % args.frame_step != 0:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]

    found, corners = cv2.findChessboardCorners(
        img, pattern_size, flags=cv2.CALIB_CB_FILTER_QUADS
    )
    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        used_frames += 1
        img_pts.append(corners.reshape(1, -1, 2))
        obj_pts.append(pattern_pts.reshape(1, -1, 3))

        print(f"[✅] -- Checkerboard pattern in frame {frame}")

        if args.max_frames is not None and used_frames >= args.max_frames:
            print(f"Found {used_frames} frames with the chessboard.")
            break
    else:
        print(f"[❌] -- Checkerboard pattern in frame {frame}")

        if args.debug_dir:
            img_chess = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(img_chess, pattern_size, corners, found)
            cv2.imwrite(os.path.join(args.debug_dir, "%04d.png" % frame), img_chess)

print("Calibrating camera...")
rms, K, d, rvecs, tvecs = cv2.calibrateCamera(
    obj_pts, img_pts, (width, height), None, None
)
print("RMS:", rms)
print("K:\n", K)
print("d:", d.ravel())

K_ = K.tolist()
d_ = d.tolist()[0]

export_intrinsics_to_xml(K_, d_, args.output)
