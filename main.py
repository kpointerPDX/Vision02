import cv2 as cv                                                                                                        # import openCV-contrib-python
import numpy as np                                                                                                      # import numpy

#Global parameters:
FEED_RES_W = 1080                                                                                                       # Video feed resolution width
FEED_RES_H = 608                                                                                                        # Video feed resolution height
LOWER_HSV1 = np.array([0, 128, 32])                                                                                     # range values specifying HSV masks
UPPER_HSV1 = np.array([12, 255, 255])
LOWER_HSV2 = np.array([164, 96, 32])
UPPER_HSV2 = np.array([180, 255, 255])
ARC_LENGTH_THRESHOLD = 350                                                                                              # Arc length threshold for filtering
TRACE_ROUGHNESS = 40                                                                                                    # Allowed error from curve for polygon
STAGNATION_LIMIT = 15                                                                                                   # Frames to "hold" last detected shape

if __name__ == '__main__':                                                                                              # run only if not being imported
    cam = cv.VideoCapture(0)                                                                                            # instantiate cam feed
    cam.set(3, FEED_RES_W)                                                                                              # set feed resolution
    cam.set(4, FEED_RES_H)
    previousRect = (0, 0, 0, 0)                                                                                         # storage to remember rectangle
    stagnation = 0                                                                                                      # counter for bridging frame gaps
    feedRunning = True                                                                                                  # boolean loop variable
    while feedRunning:
        ret, frame = cam.read()                                                                                         # frame = source video frame image
        ARframe = frame.copy()                                                                                          # copy frame for final output
        hsvFrame = cv.GaussianBlur(frame, (11, 11), cv.BORDER_DEFAULT)                                                  # blur first...
        hsvFrame = cv.cvtColor(hsvFrame, cv.COLOR_BGR2HSV)                                                              # ...then convert to HSV
        mask1 = cv.inRange(hsvFrame, LOWER_HSV1, UPPER_HSV1)                                                            # HSV mask to cover the bottom hues
        mask2 = cv.inRange(hsvFrame, LOWER_HSV2, UPPER_HSV2)                                                            # HSV mask to cover the top hues
        mask = cv.bitwise_or(mask1, mask2)
        masked = cv.bitwise_and(frame, frame, mask=mask)                                                                # masked frame image
        masked = cv.GaussianBlur(masked, (11, 11), cv.BORDER_DEFAULT)                                                   # blur to reduce small edges/noise
        boundaries = cv.Canny(masked, 40, 60)                                                                           # edge detection
        contours, hierarchies = cv.findContours(boundaries, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)                       # convert edges to contours
        contourField = np.zeros(frame.shape, dtype="uint8")                                                             # new canvas for contours
        cv.drawContours(contourField, contours, -1, (0, 0, 255), 2)                                                     # draw contours
        traceField = np.zeros(frame.shape, dtype="uint8")                                                               # new canvas for isolated shapes
        located = 0                                                                                                     # found shapes counter
        for contour in contours:
            L = cv.arcLength(contour, True)                                                                             # arc length of current contour
            if L >= ARC_LENGTH_THRESHOLD:                                                                               # ignore small contours
                trace = cv.approxPolyDP(contour, TRACE_ROUGHNESS, True)                                                 # approximate shape from contour
                x, y, w, h, = cv.boundingRect(trace)                                                                    # get rectangle parameters
                if (3 <= len(trace) <= 4):                                                                              # ignore shapes without 3-4 sides
                    located += 1
                    cv.drawContours(traceField, [trace], 0, (255, 255, 255), 3)                                         # draw shape to shape canvas
                    x, y, w, h, = cv.boundingRect(trace)                                                                # get rectangle parameters
                    previousRect = (x, y, w, h)                                                                         # save good rectangle parameters
                    cv.putText(traceField, str(L), (x + w, y + h), cv.FONT_HERSHEY_COMPLEX_SMALL,
                               1.0, (255, 255, 255), 2)                                                                 # put label on shape canvas
                    cv.rectangle(ARframe, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)                             # draw bounding rectangle on output
                    cv.rectangle(ARframe, (x - 1, y + h), (x + w + 1, y + h + 20), (0, 0, 255), thickness=-1)           # draw 'name tab' rectangle on output
                    cv.putText(ARframe, "pyramid", (x+2, y+h+12), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 0), 2)     # put label in 'name tab' on output
        if located == 0:
            stagnation += 1                                                                                             # if no shapes found, increment counter
            if stagnation < STAGNATION_LIMIT:
                x, y, w, h, = previousRect                                                                              # if within limit, use last good rect
                f = max(0, 255 - int(stagnation * (255/STAGNATION_LIMIT)))                                              # 'fading' font color for shape canvas
                cv.putText(traceField, "???", (x + w, y + h), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (f, f, f), 2)         # put label at last good location
                cv.rectangle(ARframe, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)                                 # redraw last good rectangles on output
                cv.rectangle(ARframe, (x - 1, y + h), (x + w + 1, y + h + 20), (0, 0, 255), thickness=-1)
                cv.putText(ARframe, "pyramid", (x + 2, y + h + 12), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 0), 2)   # redraw last good label on output
        else:
            stagnation = 0                                                                                              # if >0 shapes found, reset counter

        #Uncomment to show images of specific intermediate processing steps:
        # cv.imshow("source", frame)                                                                                      # raw source image
        # cv.imshow("mask", mask)                                                                                         # HSV mask
        # cv.imshow("masked", masked)                                                                                     # masked source image
        # cv.imshow("boundaries", boundaries)                                                                             # detected edges (Canny)
        cv.imshow("contours", contourField)                                                                             # raw contours
        cv.imshow("shapes", traceField)                                                                                 # isolated shape tracing
        cv.imshow("goal locator", ARframe)                                                                              # final "augmented reality" output

        if cv.waitKey(1) & 0xFF == ord('q'):                                                                            # delay 1 ms for input; is 'q' pressed?
            feedRunning = False                                                                                         # if so, kill loop variable

    cam.release()                                                                                                       # safely stop video feed
    cv.destroyAllWindows()                                                                                              # close all opened windows
