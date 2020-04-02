import cv2
import numpy as np

# cap = cv2.VideoCapture('http://192.168.35.10:8080/video?x.mjpeg')
cap = cv2.VideoCapture('imgs/test.mp4')

# load source image or video
# src_img = cv2.imread('imgs/new_scenery.jpg')
src_cap = cv2.VideoCapture('imgs/01.mp4')

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

prev_markers = {}

while cap.isOpened():
  ret, frame = cap.read()
  src_ret, src_img = src_cap.read()

  if not ret:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    break

  # rewind source video
  if not src_ret:
    src_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    src_ret, src_img = src_cap.read()

  frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
  frame_copy = frame.copy()
  result = frame.copy()

  # detect markers
  marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
    frame,
    dictionary,
    parameters=parameters
  )

  if marker_ids is not None:
    marker_ids = marker_ids.reshape((-1,))
    marker_corners = np.array(marker_corners, dtype=np.int).reshape((-1, len(marker_ids), 2))

    for marker_id, marker_corner in zip(marker_ids, marker_corners):
      cv2.polylines(frame_copy, pts=[marker_corner], isClosed=True, color=(0, 0, 255), thickness=3)
      cv2.putText(frame_copy, str(marker_id), org=tuple(marker_corner[0]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255), thickness=3)

      prev_markers[marker_id] = marker_corner

  if len(prev_markers) == 4:
    # simple math
    top_left_x, top_left_y = np.min(prev_markers[25], axis=0)

    top_right_x, _ = np.max(prev_markers[33], axis=0)
    _, top_right_y = np.min(prev_markers[33], axis=0)

    bottom_right_x, bottom_right_y = np.max(prev_markers[30], axis=0)

    bottom_left_x, _ = np.min(prev_markers[23], axis=0)
    _, bottom_left_y = np.max(prev_markers[23], axis=0)

    # add margin for better looking
    margin = (top_right_x - top_left_x) * 0.01

    dst_pts = np.array([
        [top_left_x - margin, top_left_y - margin],
        [top_right_x + margin, top_right_y - margin],
        [bottom_right_x + margin, bottom_right_y + margin],
        [bottom_left_x - margin, bottom_left_y + margin],
    ], dtype=np.int)

    cv2.polylines(frame_copy, pts=[dst_pts], isClosed=True, color=(0, 0, 255), thickness=3)

    # perspective transform
    src_np = np.array([
      [0, 0],
      [src_img.shape[1], 0],
      [src_img.shape[1], src_img.shape[0]],
      [0, src_img.shape[0]]
    ], dtype=np.float32)

    dst_np = dst_pts.astype(np.float32)

    M = cv2.getPerspectiveTransform(src=src_np, dst=dst_np)
    result_pt = cv2.warpPerspective(src_img, M=M, dsize=(frame.shape[1], frame.shape[0]))

    # merge images
    cv2.fillPoly(result, pts=[dst_pts], color=0, lineType=cv2.LINE_AA)

    result = result + result_pt

  # visualize
  cv2.imshow('frame', frame_copy)
  cv2.imshow('result', result)

  if cv2.waitKey(1) == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
