import numpy as np
import cv2

def red_mask(hsvFrame, kernel, imageFrame):
    red_lower = np.array([136, 87, 111], np.uint8) 
    red_upper = np.array([180, 255, 255], np.uint8) 
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 
    red_mask = cv2.dilate(red_mask, kernel) 
    res_red = cv2.bitwise_and(imageFrame, imageFrame, mask=red_mask)
    return res_red

def green_mask(hsvFrame, kernel, imageFrame):
    green_lower = np.array([25, 52, 72], np.uint8) 
    green_upper = np.array([102, 255, 255], np.uint8) 
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
    green_mask = cv2.dilate(green_mask, kernel) 
    res_green = cv2.bitwise_and(imageFrame, imageFrame, mask=green_mask) 
    return res_green

def blue_mask(hsvFrame, kernel, imageFrame):
    blue_lower = np.array([94, 80, 2], np.uint8) 
    blue_upper = np.array([120, 255, 255], np.uint8) 
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 
    blue_mask = cv2.dilate(blue_mask, kernel) 
    res_blue = cv2.bitwise_and(imageFrame, imageFrame, mask=blue_mask) 
    return res_blue

webcam = cv2.VideoCapture(0)

while True:
    _, imageFrame = webcam.read()

    hsv_frame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), dtype=np.uint8)

    color_functions = [
        {"func": red_mask, "name": "Red Colour" ,"cont_color":"(0, 0, 255)"},
        {"func": green_mask, "name": "Green Colour","cont_color":"(0, 255, 0)"},
        {"func": blue_mask, "name": "Blue Colour","cont_color":"(255, 0, 0)"}
    ]

    for color_func_info in color_functions:
        
        res_color = color_func_info["func"](hsv_frame, kernel, imageFrame)
        res_color_gray = cv2.cvtColor(res_color, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(res_color_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 300:
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), eval(color_func_info["cont_color"]), 2)
                cv2.putText(imageFrame, color_func_info["name"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,eval(color_func_info["cont_color"]) )

    cv2.imshow("Multiple Color Detection in Real-Time", imageFrame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break