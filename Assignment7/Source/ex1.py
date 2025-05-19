import cv2
import numpy as np

image_pts = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(image_pts) < 10:
        image_pts.append((x, y))
        print(f"Point {len(image_pts)}: ({x}, {y})")
        if len(image_pts) == 10:
            cv2.destroyAllWindows()

# Load your image
img = cv2.imread('../Assets/house1.png')
if img is None:
    raise FileNotFoundError("Could not load house1.png")

# Create window and set callback
cv2.namedWindow('Click 10 points', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('Click 10 points', click_event)

# Display image and collect 10 points
while True:
    cv2.imshow('Click 10 points', img)
    # Small wait to process GUI events
    key = cv2.waitKey(1) & 0xFF
    if len(image_pts) >= 10 or key == ord('q'):
        # Close window and give GUI time to update
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        break

image_pts = np.array(image_pts, dtype=float)
print("Selected image points (x, y):")
print(image_pts)