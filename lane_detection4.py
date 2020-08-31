import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import copy
import os

def colorfilter(img):
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)

    lower = np.array([0,190,0])
    upper = np.array([255,255,255])
    yellower = np.array([10,0,90])
    yelupper = np.array([50,255,255])

    white = cv2.inRange(hls,lower,upper)
    yellow = cv2.inRange(hls,yellower,yelupper)

    masked = cv2.bitwise_or(white,yellow)
    img = cv2.bitwise_and(img,img,mask = masked)

    return img

def region_of_interest(img):
    x = img.shape[1]
    y = img.shape[0]

    shape = np.array([
        [int(0),int(y)],
        [int(x),int(y)],
        [int(0.55*x),int(0.6*y)],
        [int(0.45*x),int(0.6*y)],
    ])

    if len(img.shape) > 2:
        channel = img.shape[2]
        color = (255,)*channel
    else:
        color = 255

    copy_img = np.zeros_like(img)
    cv2.fillPoly(copy_img,np.int32([shape]),color,cv2.LINE_AA)

    masked = cv2.bitwise_and(copy_img,img)

    return masked

def edge_detection(img,th1,th2):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray,th1,th2)

    return edge

def line_coordiantes(img,minlength,maxgap):
    lines = cv2.HoughLinesP(
        img,
        rho = 1,
        theta = np.pi/180,
        threshold = 20,
        lines = np.array([]),
        minLineLength = minlength,
        maxLineGap = maxgap
    )

    return lines

def average_slope_intercept(img,lines):
    left_lane = []
    left_weight = []
    right_lane = []
    right_weight = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            if x1 == x2:
                continue

            slope = (y1-y2)/(x1-x2)
            intercept = (y1 - slope*x1)
            length = np.sqrt((y2-y2)**2 + (x2-x1)**2)

            if slope < 0:
                left_lane.append((slope,intercept))
                left_weight.append(length)

            else:
                right_lane.append((slope,intercept))
                right_weight.append(length)

    left_lane = np.dot(left_weight,left_lane)/np.sum(left_weight) if len(left_weight) > 0 else None
    right_lane = np.dot(right_weight,right_lane)/np.sum(right_weight) if len(right_weight) > 0 else None

    return left_lane,right_lane

def make_pixel_coordinates(lines,y1,y2):
    if lines is None:
        return
    
    slope,intercept = lines

    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)

    return ((x1,y1),(x2,y2))

def lane_lines(img,lines):
    left_lane, right_lane = average_slope_intercept(img,lines)

    y1 = img.shape[0]
    y2 = 0.6*y1

    left_line = make_pixel_coordinates(left_lane, y1, y2)
    right_line = make_pixel_coordinates(right_lane, y1, y2)

    return left_line, right_line

def draw_lines(img,lines):
    if lines is None:
        return

    line_img = np.zeros_like(img,dtype = np.uint8)

    lanes = np.array(lane_lines(img,lines))

    for point in lanes:
        cv2.line(line_img, (point[0,0],point[0,1]), (point[1,0],point[1,1]),(0,0,255),20,cv2.LINE_AA)

    shape = []
    for line in lanes:
        shape.append(list(line))

    shape = np.array(shape)
    roi_points = np.array([
        shape[0,0],
        shape[1,0],
        shape[1,1],
        shape[0,1]

    ])
    
    cv2.fillPoly(line_img,np.int32([roi_points]),(0,255,0))
    img = cv2.addWeighted(img,0.8,line_img,0.4,0.0)

    return img

# image = Image.open("./lane_1.png").convert("RBG")
img = cv2.imread("C:/Users/Rachit/Onedrive/Desktop/Files/VS/Detection/Lane/lane_1.png")

RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
hls_colorfilter = colorfilter(RGB)
roi_img = region_of_interest(hls_colorfilter)
edge_img = edge_detection(roi_img,50,100)
edge_coordinates = line_coordiantes(edge_img,20,300)

x = draw_lines(img,edge_coordinates)

cv2.imshow("Frame",x)
cv2.waitKey(0)