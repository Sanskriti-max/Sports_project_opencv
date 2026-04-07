import cv2
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

video_path = r"C:\Users\VIBHA KUMARI\Downloads\19345257-uhd_3840_2160_30fps.mp4"  # 🔴 change if needed

if not os.path.exists(video_path):
    print("❌ ERROR: Video file not found")
    exit()


model = YOLO("yolov8n.pt")   # ⚡ faster model
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ ERROR: Video not opening")
    exit()
else:
    print("✅ Video opened successfully")


player_class = 0
ball_class = 32

player_ids = set()
ball_positions = []

frame_count = 0   # 🔥 for skipping frames

# Window size control
cv2.namedWindow("Cricket Analysis", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Cricket Analysis", 600, 350)

# -------------------
# MAIN LOOP
# -------------------
while True:
    ret, frame = cap.read()

    if not ret:
        print("✅ Video finished")
        break

    frame_count += 1

    # 🔥 Skip every alternate frame (speed boost)
    if frame_count % 2 != 0:
        continue

    # Resize for speed
    frame = cv2.resize(frame, (480, 270))

    results = model(frame)
    detections = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, classes):

            x1, y1, x2, y2 = box

            # Player detection
            if int(cls) == player_class:
                detections.append(([x1, y1, x2-x1, y2-y1], score, 'player'))

            # Ball detection (improved)
            if int(cls) == ball_class and score > 0.3:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                ball_positions.append((cx, cy))

                # Draw ball
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, "Ball", (cx, cy-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # Tracking players
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        player_ids.add(track_id)

        l, t, r, b = track.to_ltrb()

        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0,255,0), 2)
        cv2.putText(frame, f"Player {track_id}",
                    (int(l), int(t)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Limit ball trail
    ball_positions = ball_positions[-30:]

    for pos in ball_positions:
        cv2.circle(frame, pos, 3, (0,0,255), -1)

    # Display info
    cv2.putText(frame, f"Players: {len(player_ids)}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    cv2.imshow("Cricket Analysis", frame)

    # Faster refresh
    if cv2.waitKey(1) & 0xFF == 27:
        break

# CLEANUP

cap.release()
cv2.destroyAllWindows()