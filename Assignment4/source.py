import numpy as np
import cv2
import matplotlib.pyplot as plt

# Create a synthetic color image with size 100x200 pixels
height, width = 100, 200
# Initialize an image with 3 channels (B, G, R) set to zero
img = np.zeros((height, width, 3), dtype=np.uint8)

# Set up a vertical edge:
# - For the red channel (index 2 in OpenCV), left half is bright and right half is dark.
img[:, :width//2, 2] = 255  # Red channel: white on left, 0 on right

# - For the green channel (index 1), left half is dark and right half is bright.
img[:, width//2:, 1] = 255  # Green channel: 0 on left, white on right

# Blue channel (index 0) remains 0 throughout.

# Define the Sobel kernels for x and y directions
gx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float32)
gy = gx.T # swaps rows and columns of gx

# Function to compute the x and y gradients using cv2.filter2D
def compute_gradients(channel):
    grad_x = cv2.filter2D(channel.astype(np.float32), -1, gx)
    grad_y = cv2.filter2D(channel.astype(np.float32), -1, gy)
    return grad_x, grad_y

# Split the image into B, G, R channels (note: OpenCV uses BGR order)
B, G, R = cv2.split(img)

# Compute gradients for each channel
Rx, Ry = compute_gradients(R)
Gx, Gy = compute_gradients(G)
Bx, By = compute_gradients(B)  # This will be zero everywhere

# Sum the gradients of the three channels as described in the problem
grad_x_total = Rx + Gx + Bx
grad_y_total = Ry + Gy + By

# Compute the overall gradient magnitude
grad_magnitude = np.sqrt(grad_x_total**2 + grad_y_total**2)

# Display the synthetic image and the resulting gradients
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
# Convert BGR to RGB for correct display in matplotlib
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Synthetic Color Image")
plt.axis('off')
cv2.imwrite(r"Images/synthetic_image.png", img)

plt.subplot(2, 2, 2)
plt.imshow(grad_x_total, cmap='gray')
plt.title("Summed Gradient X")
plt.axis('off')
cv2.imwrite(r"Images/gradient_x.png", grad_x_total)

plt.subplot(2, 2, 3)
plt.imshow(grad_y_total, cmap='gray')
plt.title("Summed Gradient Y")
plt.axis('off')
cv2.imwrite(r"Images/gradient_y.png", grad_y_total)

plt.subplot(2, 2, 4)
plt.imshow(grad_magnitude, cmap='gray')
plt.title("Summed Gradient Magnitude")
plt.axis('off')
cv2.imwrite(r"Images/gradient_magnitude.png", grad_magnitude)

plt.tight_layout()
plt.show()
