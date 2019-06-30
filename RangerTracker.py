# Imports
import numpy as np
import cv2 as cv
from imutils.video import VideoStream
import imutils
import time

# Variables
MIN_AREA = 500
BOUNDING_BOX_COLOR = (0, 255, 0)

# Get the video stream
vStream = VideoStream(src=0).start()
time.sleep(2.0)

# Establish first frame
firstFrame = None
gotFirstFrame = False

# Enter recording loop
while True:
    # Get the frame
    frame = vStream.read()

    # Check if video has ended
    if not frame.any():
        break

    # Set display text
    dText = "Clear"

    # Resize the frame
    frame = imutils.resize(frame, width=500)

    # Prepare comparison frame
    frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frameGray = cv.GaussianBlur(frameGray, (21, 21), 0)

    # Check if first frame established
    if not gotFirstFrame:
        # Set the first frame
        firstFrame = frameGray

        # Mark and continue
        gotFirstFrame = True
        continue
    
    # Calculate delta difference between first and current frame
    frameDelta = cv.absdiff(firstFrame, frameGray)

    # Set frame delta threshold
    threshold = cv.threshold(frameDelta, 25, 255, cv.THRESH_BINARY)[1]

    # Dialate the threshold to fill holes
    threshold = cv.dilate(threshold, None, iterations=2)

    # Calculate the contours of the threshold
    contours = cv.findContours(threshold.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Loop through found contours
    for contour in contours:
        # Check if contour is too small
        if cv.contourArea(contour) < MIN_AREA:
            continue

        # Apply contour bounding box
        (x, y, w, h) = cv.boundingRect(contour)
        cv.rectangle(frame, (x, y), (x+w, y+h), BOUNDING_BOX_COLOR, 2)

        # Change display text
        dText = "Detected"

    # Add display text to frame
    cv.putText(frame, str(dText), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display frames
    cv.imshow("Feed", frame)
    cv.imshow("Threshold", threshold)
    cv.imshow("Delta", frameDelta)

    # Look for keypress
    key = (cv.waitKey(1) & 0xFF)

    # Check if q key
    if key == ord("q"):
        break

# End the video stream
vStream.stop()

# Close all OpenCV windows
cv.destroyAllWindows()
