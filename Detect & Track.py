import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture("C:\Users\VIBHA KUMARI\Downloads\19345257-uhd_3840_2160_30fps.mp4")

player_class = 0
ball_class = 32

player_ids = set()
ball_positions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, classes):

            if int(cls) == player_class:
                x1, y1, x2, y2 = box
                detections.append(([x1, y1, x2-x1, y2-y1], score, 'player'))

            if int(cls) == ball_class:
                x1, y1, x2, y2 = box
                ball_positions.append((int(x1), int(y1)))

    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw players
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

    # Draw ball
    for pos in ball_positions:
        cv2.circle(frame, pos, 4, (0,0,255), -1)

    cv2.putText(frame, f"Players: {len(player_ids)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.imshow("Cricket Analysis", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()