# collect_faces.py
#
# Usage:
#   python collect_faces.py keanu
#
# It will:
#   - Open your webcam
#   - Detect your face with Haar
#   - Save cropped face images into dataset/<label>/img_XX.jpg

import cv2
import os
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python collect_faces.py <label_name>")
        return

    label = sys.argv[1]
    save_dir = os.path.join("dataset", label)
    os.makedirs(save_dir, exist_ok=True)

    # Load Haar cascade for faces
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        print("Error: Could not load Haar cascade.")
        return

    cap = cv2.VideoCapture(0)  # webcam
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    count = 0
    print(f"[INFO] Collecting faces for label '{label}'. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            # Draw preview box on screen
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop the face region and save it
            face_roi = gray[y:y + h, x:x + w]  # grayscale face
            img_path = os.path.join(save_dir, f"img_{count:04d}.jpg")
            cv2.imwrite(img_path, face_roi)
            count += 1

        cv2.putText(
            frame,
            f"Label: {label}  Saved: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Collecting Faces", frame)

        # Stop on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"[INFO] Saved {count} images to {save_dir}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
