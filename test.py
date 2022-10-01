import ffmpeg
import numpy
import cv2


url = 'rtsp://192.168.88.221/live'
cap = cv2.VideoCapture(url)
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Display the resulting frame
    cv2.imshow('cv2-get', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
