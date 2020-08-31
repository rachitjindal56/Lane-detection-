import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import warning 
import copy
warning.filterwarnings('ignore')

def colorfilter_hls(img):
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)

    lower = np.array([0,190,0])
    upper = np.array([255,255,255])
    ylower = np.array([10,0,90])
    yupper = np.array([50,255,255])

    white = cv2.inRange(hls,lower,upper)
    yellow = cv2.inRange(hls,ylower,yupper)

    masked = cv2.bitwise_or(white,yellow)
    masked = cv2.bitwise_and(img, img, mask=masked)

    return masked

def region_of_interest(img):
    x = img.shape[1]
    y = img.shape[0]

    shape = np.array([
        [int(0),int(y)],
        [int(x),int(y)],
        [int(x*0.55),int(y*0.6)],
        [int(x*0.45),int(y*0.6)]
    ])

    if len(img.shape) > 2:
        channel = img.shape[2]
        color = (255,)*channel
    else:
        color = 255

    mask = np.zeros_like(img)
    cv2.fillPoly(mask,np.int32([shape]),color)

    masked = cv2.bitwise_and(img,mask)
    return masked

def edge_detection(img,th1,th2):
    gray_scale = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    edge_canny = cv2.Canny(gray_scale, th1, th2)

    return edge_canny

def edge_coordinates(img,minlength,maxgap):
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
    left_lane   = []
    left_weight  = []
    right_lane  = []
    right_weight = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            if x1 == x2:
                continue

            slope = (y1-y2)/(x1-x2)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2 + (x2-x1)**2)

            if slope < 0:
                left_lane.append((slope,intercept))
                left_weight.append(length)
            else:
                right_lane.append((slope,intercept))
                right_weight.append(length)
    
    left_lane = np.dot(left_weight, left_lane)/np.sum(left_weight) if len(left_weight) > 0 else None
    right_lane = np.dot(right_weight, right_lane)/np.sum(right_weight) if len(right_weight) > 0 else None

    print(left_lane)

    return left_lane,right_lane
    
def make_pixel_coordinates(lines,y1,y2):
    if lines is None:
        return
    
    slope, intercept = lines

    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)

    return ((x1,y1),(x2,y2))

def lane_lines(img,lines):
    left_lane, right_lane = average_slope_intercept(img,lines)

    y1 = img.shape[0]
    y2 = y1*0.6

    left_line  = make_pixel_coordinates(left_lane, y1, y2)
    right_line = make_pixel_coordinates(right_lane, y1, y2)
    
    return left_line, right_line

def draw_lines(img,lines):
    if lines is None:
        return

    line_img = np.zeros_like(img,dtype=np.uint8)

    lane = np.array(lane_lines(img,lines))
    for line in lane:
        cv2.line(line_img, (line[0,0],line[0,1]), (line[1,0],line[1,1]), (255,0,0),10)

    shape = []
    for line in lane:
        shape.append(list(line))

    shape = np.array(shape)
    shaped = np.array([
        shape[0,1],
        shape[1,1],
        shape[1,0],
        shape[0,0]
    ])

    cv2.fillPoly(line_img, np.int32([shaped]), color = (100,238,100))
    masked = cv2.addWeighted(img, 0.8, line_img, 0.4, 0.0)

    return masked

# img = Image.open("C:/Users/Rachit/Onedrive/Desktop/Files/VS/Detection/lane_2.png").convert("RGB")
image = cv2.imread("C:/Users/Rachit/Onedrive/Desktop/Files/VS/Detection/Lane/lane_1.png")
RGB_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

filtered_img = colorfilter_hls(RGB_image)
cropped_img = region_of_interest(filtered_img)
edge_img = edge_detection(cropped_img, 50,100)

edge_points = edge_coordinates(edge_img,20,300)

# average_slope_intercept(image,edge_points)
lines_image = draw_lines(RGB_image,edge_points)
image = cv2.cvtColor(lines_image, cv2.COLOR_RGB2BGR)
# print(edge_points)

cv2.imshow("the frame",image)
cv2.waitKey(0)
