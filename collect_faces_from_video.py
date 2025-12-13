# collect_faces_from_video.py
#
# Usage:
#   python collect_faces_from_video.py keanu testvideo.mp4
#
# It will:
#   - Open the given video file
#   - Detect faces with Haar
#   - Save cropped face images into dataset/<label>/img_XXXX.jpg

import cv2
import os
import sys

TARGET_WIDTH = 960  # <- same idea as realtime_yolo.py

def main():
    # Expect: script label video_path
    if len(sys.argv) < 3:
        print("Usage: python collect_faces_from_video.py <label_name> <video_path>")
        return

    label = sys.argv[1]
    video_path = sys.argv[2]

    save_dir = os.path.join("dataset", label)
    os.makedirs(save_dir, exist_ok=True)

    # Load Haar cascade for faces
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        print("Error: Could not load Haar cascade.")
        return

    # Open the video file instead of camera
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    count = 0
    print(f"[INFO] Collecting faces for label '{label}' from {video_path}. Press 'q' to stop early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Reached end of video.")
            break

        # --- Resize frame like realtime_yolo.py ---
        h, w = frame.shape[:2]
        if w > TARGET_WIDTH:
            scale = TARGET_WIDTH / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # ------------------------------------------

        # Convert frame to grayscale for Haar
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the whole frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w_face, h_face) in faces:
            # Draw preview box on screen
            cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), (0, 255, 0), 2)

            # Crop the face region and save it (grayscale)
            face_roi = gray[y:y + h_face, x:x + w_face]
            img_path = os.path.join(save_dir, f"img_{count:04d}.jpg")
            cv2.imwrite(img_path, face_roi)
            count += 1

        # HUD text
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

        cv2.imshow("Collecting Faces from Video", frame)

        # Stop early on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"[INFO] Saved {count} images to {save_dir}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
