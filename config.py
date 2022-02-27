import numpy as np

# [ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ]
cameramtx = np.array([[481.32639529,   0,         327.59856174],
    [  0,         479.64805952, 261.18746562],
    [  0,           0,           1        ]])
dist = np.array([[-4.37450682e-01,  3.84366023e-01, -6.00142834e-04, -3.12503696e-04,
  -3.02112133e-01]])


# Constants for drawing on image
line_thickness = 1
center_line_length = 3
text_offset = 5
color = (255, 255, 255)