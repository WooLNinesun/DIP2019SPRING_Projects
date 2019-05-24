import numpy as np
import cv2
from matplotlib import pyplot as plt
from util import *


im = cv2.cvtColor(read_raw_image('raw/sample1.raw', 256, 256), cv2.COLOR_GRAY2BGR)
cv2.imwrite('output/sample2.jpg', im)
