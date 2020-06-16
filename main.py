import cv2
import numpy as np

frame_height = 640
frame_width = 480

# Capture using webcam
cap = cv2.VideoCapture(0)
# Set width and height of frame
cap.set(3, frame_height) # First param is Property Identifer for the video
# Read here: https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
# Or refer propID.txt
cap.set(4, frame_width)
# Set the brightness
cap.set(10, 140)


# Colors which we want to detect
colors = [[94,105,2,121,255,255],  # Blue # Order is: HueMin, SatMin, ValMin, HueMax, SatMax, ValMax
            [0,133,100,179,255,255],  # Red
            [25,52,60,102,255,255]]  # Green


drawingColors = [[204, 102, 0],   # Blue # Using BGR
                 [0, 0, 204],   # Red
                 [0, 128, 0]]  # Green


drawingPoints = [] # x, y, drawingColorIndex

def get_contours(img):
    contours, heirarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # image, retrieval method, approximation-(where you can request for all info or we can request for compressed info)
    # RETR_EXTERNAL retrieves extremes outer contours, there are other retrieval methods too
    x, y, w, h = 0, 0, 0, 0
    # contours are saved in contours variable above
    for cnt in contours:
        area = cv2.contourArea(cnt) # Find area of contour
        # Now to neglect smaller shapes, if any
        if area > 500:
            # # This will draw a blue line on the imgFinal image
            # cv2.drawContours(imgFinal, cnt, -1, (255, 0, 0), 3) # Draw the contours, params: img, contour, contour_index(-1 means all the contours), color, thickness
            # Now we will calculate curve length, curve length will help us approximate corners of our shape
            peri = cv2.arcLength(cnt, True)  # params: curve, closed or not
            corner_points = cv2.approxPolyDP(cnt, 0.02*peri, True) # contour, resolution(play around with to get good results), closed or not
            # Now we create bounding box around detected object
            # To draw bounding box, we need x and y and also width and height
            x, y, w, h = cv2.boundingRect(corner_points)
    # Now we want to draw from tip of object
    return x+w//2, y # Center coordinate and top point z


def drawOnCanvas(drawingPoints):
    for point in drawingPoints:
        cv2.circle(imgFinal, (point[0], point[1]), 7, drawingColors[point[2]], cv2.FILLED) # params: image, center points, radius, color, thickness

def detect_color(img, drawingColors):
    """
    Here we don't want to detect single color, we want to find all different types of color
    """
    # In HSV, it is easier to represent a color than in BGR color-space
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    counter = 0
    new_points = []
    # Looping to detect all the colors mentioned in list
    for color in colors:
        # Now we will used these values to filter out image, so we can get image in that particular color in that range
        lower_limit = np.array(color[0:3]) # First three 
        upper_limit = np.array(color[3:6]) # Last three
        # This will filter out
        mask = cv2.inRange(imgHSV, lower_limit, upper_limit)
        # cv2.imshow(str(color[0]), mask) # As we can not have a generic name so we for now we just write Hue Min Value
        # Now we want to draw a binding box around the captured color
        x, y = get_contours(mask)
        cv2.circle(imgFinal, (x, y), 7, drawingColors[counter], cv2.FILLED) # params: image, center points, radius, color, thickness
        if x and y:
            new_points.append([x,y, counter])
        counter += 1
    return new_points


# Video is just a sequence of images, so we need while loop to go through each frame
while True:
    ret, frame = cap.read()
    imgFinal = frame.copy()
    new_points = detect_color(frame, drawingColors)
    if new_points:
        for point in new_points:
            drawingPoints.append(point)
    if drawingPoints:
        drawOnCanvas(drawingPoints)
    cv2.imshow("Result", imgFinal)
    # Now we need to detect color

    # Now add delay and also add `q` button press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
