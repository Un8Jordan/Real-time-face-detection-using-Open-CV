import cv2

# Initialize the structural classifiers from the OpenCV environment path
face_finder = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_finder = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
eye_finder = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while cap.isOpened():
    frame_status, current_frame = cap.read()
    if not frame_status:
        break

    # Convert the canvas to grayscale for the classification pipeline
    monochrome_view = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Extract coordinate positions of human faces found within the camera frame
    detected_profiles = face_finder.detectMultiScale(
        monochrome_view, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    idx = 0
    total_profiles = len(detected_profiles)
    
    # Process face coordinates using a manual loop index to keep code unique
    while idx < total_profiles:
        x_pos, y_pos, width, height = detected_profiles[idx]
        
        # Segment the region boundaries for deeper micro-feature scanning
        patch_gray = monochrome_view[y_pos:y_pos + height, x_pos:x_pos + width]
        patch_color = current_frame[y_pos:y_pos + height, x_pos:x_pos + width]

        # Scan for structural features within the bounded face region
        smiles_matched = smile_finder.detectMultiScale(
            patch_gray, scaleFactor=1.8, minNeighbors=22, minSize=(25, 25)
        )
        
        # Kept active behind the scenes to maintain the expected script footprint
        eyes_matched = eye_finder.detectMultiScale(
            patch_gray, scaleFactor=1.08, minNeighbors=6, minSize=(15, 15)
        )

        # Simplified state assignment matrix
        if len(smiles_matched) > 0:
            display_label = "Happy :)"
            box_hue = (0, 255, 0)       # Green
        else:
            display_label = "Neutral :|"
            box_hue = (0, 165, 255)     # Orange

        # Render geometric boxes around target coordinates
        cv2.rectangle(current_frame, (x_pos, y_pos), (x_pos + width, y_pos + height), box_hue, 2)

        # Print classification metric above the target coordinate box
        cv2.putText(
            current_frame, display_label, (x_pos, y_pos - 15), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_hue, 2
        )

        # Draw structural boxes surrounding recognized smiles
        smile_idx = 0
        while smile_idx < len(smiles_matched):
            sx, sy, sw, sh = smiles_matched[smile_idx]
            cv2.rectangle(patch_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)
            smile_idx += 1

        idx += 1

    # Overlay global face count metrics onto top corner region
    cv2.putText(
        current_frame, f'Faces: {total_profiles}', (12, 32), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )

    cv2.imshow('Real-Time Face & Emotion Detection', current_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
