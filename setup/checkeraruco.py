import cv2

import numpy as np

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# 7 columns, 5 rows, 3cm squares, 1.5cm markers

board = cv2.aruco.CharucoBoard_create(7, 5, 0.03, 0.015, dictionary)

# Draw the board at high resolution

board_image = board.draw((2100, 1500))

# Add white border

canvas = np.ones((1700, 2300), dtype=np.uint8) * 255

offset_x = (2300 - 2100) // 2

offset_y = (1700 - 1500) // 2

canvas[offset_y:offset_y+1500, offset_x:offset_x+2100] = board_image

cv2.imwrite("charuco_board.png", canvas)

print("Saved")
