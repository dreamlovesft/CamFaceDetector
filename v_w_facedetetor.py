# --face cascades/haarcascade_frontalface_default.xml --video video/adrian_face.mov


from pyimagesearch.facedetector import FaceDetector
from pyimagesearch import imutils
# from picamera.array import PiRGBArray
# from picamera import PiCamera
import argparse
import time
import cv2



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required = True,
    help = "path to where the face cascade resides")
ap.add_argument("-v", "--video",
    help = "path to the (optional) video file")
args = vars(ap.parse_args())

if not args.get('video', False):
    fd = FaceDetector(args["face"])
    time.sleep(0.1)
    camera = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()
        frame = imutils.resize(frame, width = 300)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceRects = fd.detect(gray, scaleFactor = 1.1, 
        minNeighbors = 5,minSize = (30, 30))
        frameClone = frame.copy()


        for (fX, fY, fW, fH) in faceRects:
            cv2.rectangle(frameClone, (fX, fY), (fX+fW, fY+fH), (0,255,0), 2)

        cv2.imshow("Face", frameClone)

        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()    

else: 
    fd = FaceDetector(args["face"])
    camera = cv2.VideoCapture(args["video"])
    # keep looping
    while True:
        (grabbed, frame) = camera.read()
        # if we are viewing a video and we did not grab a
        # frame, then we have reached the end of the video
        if args.get("video") and not grabbed:
            break

        # resize the frame and convert it to grayscale
        frame = imutils.resize(frame, width = 300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in the image and then clone the frame
        # so that we can draw on it
        faceRects = fd.detect(gray, scaleFactor = 1.1, minNeighbors = 5,
            minSize = (30, 30))
        frameClone = frame.copy()

        # Loop over the face bounding boxes and draw them
        for (fX, fY, fW, fH) in faceRects:
            cv2.rectangle(frameClone, (fX, fY), (fX+fW, fY+fH), (0,255,0), 2)



            cv2.imshow('face', frameClone)

            if cv2.waitKey(1) & 0xFF==ord('q'):
                break


    camera.release()
    cv2.destroyAllWindows()



