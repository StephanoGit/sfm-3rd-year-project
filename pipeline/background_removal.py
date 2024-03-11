from PIL import Image
from rembg import remove
import glob

images = []

for file_name in glob.glob("*.jpg"):
    blr_img = remove(Image.open(file_name))
    blr_img.save("blr_" + file_name[:-4] + ".png")
