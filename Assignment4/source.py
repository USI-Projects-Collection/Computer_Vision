import numpy as np
import cv2
import matplotlib.pyplot as plt


################### EXERCISE 1 ###################

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

# Function to compute the x and y gradients using
def compute_gradients(channel):
    grad_x = cv2.filter2D(channel.astype(np.float32), -1, gx)
    grad_y = cv2.filter2D(channel.astype(np.float32), -1, gy)
    return grad_x, grad_y

# Split the image into B, G, R channels (apparently OpenCV uses BGR order)
B, G, R = cv2.split(img)

# Compute gradients for each channel
Rx, Ry = compute_gradients(R)
Gx, Gy = compute_gradients(G)
Bx, By = compute_gradients(B)  # This will be zero everywhere

# Sum of the gradients of the three channels
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


########################## EXERCISE 2 ##########################

def otsu_thresholding(image):
    # Step 1: Convert the image to grayscale if it's not already
    if len(image.shape) == 3:  # Check if the image is colored (RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # Step 2: Apply Otsu's Thresholding
    # Otsu's method automatically calculates the optimal threshold
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary_image

# Load the image (houses.pgm)
image = cv2.imread('Assignment4/houses.pgm')

# Step 1: Apply Otsu's method to find the optimal threshold
binary_image = otsu_thresholding(image)

# Step 2: Display the original and binary image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert to RGB for matplotlib display
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Otsu Thresholding')
cv2.imwrite(r"Images/otsu_thresholding.png", binary_image)
plt.show()