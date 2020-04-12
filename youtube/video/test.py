import numpy as np
import cv2

def get_cameras():
	index = 0
	arr = []
	while True:
    		cap = cv2.VideoCapture(index)
    		if not cap.read()[0]:
        		break
    		else:
        		arr.append(index)
    		cap.release()
    		index += 1
	return arr


cap = cv2.VideoCapture(0)

while(True):
    print("*")
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
