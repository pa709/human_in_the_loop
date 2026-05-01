
import cv2
import numpy as np

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker = cv2.aruco.drawMarker(dictionary, 0, 800)

white_canvas = np.ones((1000, 1000), dtype=np.uint8) * 255
offset = (1000 - 800) // 2
white_canvas[offset:offset+800, offset:offset+800] = marker

cv2.imwrite("aruco_marker_id0.png", white_canvas)
print("Saved")



import cv2
import numpy as np

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker = cv2.aruco.drawMarker(dictionary, 0, 591)

# A4 canvas at 300 DPI
canvas = np.ones((3508, 2480), dtype=np.uint8) * 255

# Center the marker on the canvas
offset_x = (2480 - 591) // 2
offset_y = (3508 - 591) // 2
canvas[offset_y:offset_y+591, offset_x:offset_x+591] = marker

cv2.imwrite("aruco_marker_A4.png", canvas)
print("Saved")




