import cv2

# lOad Face Detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Open Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break    

    # Show Video
    cv2.imshow('Face Detector', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()