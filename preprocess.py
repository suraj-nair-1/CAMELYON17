import openslide
import numpy as np
import skimage
from skimage.viewer import ImageViewer
import PIL
import matplotlib.pyplot as plt

slide = openslide.open_slide("../CAMELYON/data/patient_013/patient_013_node_0.tif")

slide
level = 6
dims = slide.level_dimensions[level]

slide.level_downsamples[level]

pixelarray = np.array(slide.read_region((0,0), level, dims))

pixelarray.shape

viewer = ImageViewer(pixelarray)
viewer.show()