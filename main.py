import cv2
import numpy as np
import configparser

from utils import to_int, COLORS_HSV_VALUES, DRAWING_COLORS


drawingPoints = []  # x-coordinate, y-coordinate, drawingColorIndex in `DRAWING_COLORS` list


def get_contours_top_center_point(img):
    """
    Returns top center coordinates of the detected contour
    Optionally can show contour too
    """    
    contours, heirarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # image, retrieval method, approximation-(where you can request for all info or we can request for compressed info)
    # RETR_EXTERNAL retrieves extremes outer contours, there are other retrieval methods too
    x, y, w, h = 0, 0, 0, 0
    # contours are saved in contours variable above
    for cnt in contours:
        area = cv2.contourArea(cnt) # Find area of contour
        # Now to neglect smaller shapes, if any
        if area > 500:
            # This will draw a blue line on the imgFinal image
            # cv2.drawContours(imgFinal, cnt, -1, (255, 0, 0), 3) # Draw the contours, params: img, contour, contour_index(-1 means all the contours), color, thickness

            # Now we will calculate curve length, it will help us approximate corners of our shape
            peri = cv2.arcLength(cnt, True)  # params: curve, closed or not
            corner_points = cv2.approxPolyDP(cnt, 0.02*peri, True) # contour, resolution(play around with to get good results), closed or not
            # Now we create bounding box around detected object
            # To draw bounding box, we need x and y and also width and height
            x, y, w, h = cv2.boundingRect(corner_points)
    # Now we want to draw from tip of object
    return x+w//2, y # Center coordinate and top point


def paintCanvas(drawingPoints, imgFinal):
    """
    Paints the image when the color is moved on the screen
    Looping is done as more than one can color can be present
    """ 
    for point in drawingPoints:
        cv2.circle(imgFinal, (point[0], point[1]), 7, DRAWING_COLORS[point[2]], cv2.FILLED) # params: image, center points, radius, color, thickness


def detect_color(img, imgFinal):
    """
    Here more than single color can be detected
    :return coordinates and index of drawingColor
    """
    # Convert to HSV, as it is easier to represent a color in it than in BGR color-space
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    counter = 0
    # To store current color details
    curr_points = []
    # Looping to detect all the colors mentioned in list
    for color in COLORS_HSV_VALUES:
        # Now we will used these values to filter out image, so we can get image in that particular color in that range
        lower_limit = np.array(color[0:3]) # First three 
        upper_limit = np.array(color[3:6]) # Last three
        # This will filter out
        mask = cv2.inRange(imgHSV, lower_limit, upper_limit)
        # cv2.imshow(str(color[0]), mask) # As we can not have a generic name so we for now we just write Hue Min Value
        # Now we can also draw a boundary around the captured color
        x, y = get_contours_top_center_point(mask)
        # Draw a circle on the center on the top of the contour
        cv2.circle(imgFinal, (x, y), 7, DRAWING_COLORS[counter], cv2.FILLED) # params: image, center points, radius, color, thickness
        # If we have detected color and received it's top's center coordinates
        if x and y:
            # Append the details of the current color
            curr_points.append([x,y, counter])
        counter += 1
    return curr_points, imgFinal


def capture_and_paint():
    """
    Captures a video to perform futher operations
    Looping is done as video is just a sequence of images, so we need while loop to go through each frame
    """
    # 
    while True:
        # Read each frame
        ret, frame = cap.read()
        # Make a copy original image, on which we can draw
        imgFinal = frame.copy()
        # Now detect color shown on screen
        curr_points, imgF = detect_color(frame, imgFinal) # Returns coordinates and index of the color
        if curr_points:
            # As more than one colors can be shown on the screen
            for point in curr_points:
                drawingPoints.append(point)
        if drawingPoints:
            paintCanvas(drawingPoints, imgF)

        # Now display the painted frame
        cv2.imshow("Result", imgFinal)

        # Now add delay and also add `q` button press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    config = configparser.ConfigParser()
    configFilePath = "/home/shantanu/stuff/projects/virtual-paint/config.ini"
    # Read config file
    config.read(configFilePath)
    frameWidth = to_int(config['video']['frameWidth'])
    frameHeight = to_int(config['video']['frameWidth'])
    brightness = to_int(config['video']['brightness'])

    # Capture using webcam
    cap = cv2.VideoCapture(0)
    # Set width and height of frame
    cap.set(3, frameHeight) # First param is Property Identifer for the video
    # Read here: https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
    # Or refer propID.txt
    cap.set(4, frameWidth)
    # Set the brightness
    cap.set(10, brightness)
    # Call the processing function
    capture_and_paint()
