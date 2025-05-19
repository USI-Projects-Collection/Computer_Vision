import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# 1. Define the image coordinates from the provided "house1.png"
# Measured from the image shown, with points clearly marked as 1-10
image_points = np.array([
    [148, 389],  # Point 1 (bottom left corner)
    [430, 277],  # Point 2 (bottom middle - door left)
    [481, 277],  # Point 3 (bottom middle - door right)
    [676, 99],   # Point 4 (roof right corner)
    [786, 217],  # Point 5 (right wall, top corner)
    [513, 389],  # Point 6 (bottom right corner)
    [334, 99],   # Point 7 (roof left corner)
    [738, 338],  # Point 8 (right wall, window area)
    [280, 255],  # Point 9 (left wall, window area)
    [800, 389]   # Point 10 (bottom far right)
])

# 2. Define the world coordinates provided in coords.txt
world_points = np.array([
    [-1.0, -1.0, 0.0, 1.0],        # Point 1
    [0.015545, -1.0, 0.965257, 1.0],  # Point 2
    [0.491684, -1.0, 0.965257, 1.0],  # Point 3
    [0.0, 1.03586, 2.1009, 1.0],    # Point 4
    [1.05319, 1.03586, 1.15121, 1.0], # Point 5
    [1.22323, -1.22323, 0.0, 1.0],   # Point 6
    [0.0, -1.0, 2.1009, 1.0],       # Point 7
    [1.0, 0.234552, 0.489472, 1.0],  # Point 8
    [-1.05319, -1.0, 1.0, 1.0],     # Point 9
    [1.0, 1.0, 0.0, 1.0]            # Point 10
])

# 3. Apply the DLT algorithm to compute the projection matrix P
def dlt_algorithm(world_points, image_points):
    n = world_points.shape[0]
    A = np.zeros((2*n, 12))
    
    for i in range(n):
        X = world_points[i]  # 3D world coordinates (homogeneous)
        x = image_points[i]  # 2D image coordinates
        
        # First line for x coordinate
        A[2*i, 0:4] = X
        A[2*i, 8:12] = -x[0] * X
        
        # Second line for y coordinate
        A[2*i+1, 4:8] = X
        A[2*i+1, 8:12] = -x[1] * X
    
    # Solve Ap=0 using SVD
    U, S, Vt = linalg.svd(A)
    
    # p is the last column of V (last row of V transpose)
    p = Vt.T[:, -1]
    
    # Reshape into 3x4 projection matrix
    P = p.reshape(3, 4)
    
    return P

# 4. Decompose the projection matrix to get K, R, and C
def decompose_projection_matrix(P):
    # Extract the first 3x3 submatrix
    M = P[:, 0:3]
    
    # Use RQ decomposition to get K and R
    K, R = linalg.rq(M)
    
    # Make diagonal elements of K positive
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R
    
    # Ensure determinant of R is positive (proper rotation matrix)
    if np.linalg.det(R) < 0:
        K = K @ np.diag([1, 1, -1])
        R = R @ np.diag([1, 1, -1])
    
    # Normalize K
    K = K / K[2, 2]
    
    # Compute camera center C
    # P = K[R|-RC]
    # So, P = [M|p4] and p4 = -M*C
    # Therefore, C = -inv(M)*p4
    p4 = P[:, 3]
    C = -np.linalg.inv(M) @ p4
    
    return K, R, C

# Execute the DLT algorithm
P = dlt_algorithm(world_points, image_points)
print("Projection Matrix P:")
print(P)

# Decompose P to get K, R, and C
K, R, C = decompose_projection_matrix(P)

print("\nCalibration Matrix K:")
print(K)
print("\nRotation Matrix R:")
print(R)
print("\nCamera Center C:")
print(C)

# Optional: Verify the decomposition by reconstructing P
P_recon = K @ np.hstack((R, -R @ C.reshape(3, 1)))
print("\nReconstructed Projection Matrix:")
print(P_recon)

# Optional: Normalize P for comparison
P_norm = P / P[2, 3]
P_recon_norm = P_recon / P_recon[2, 3]
print("\nNormalized Original P:")
print(P_norm)
print("\nNormalized Reconstructed P:")
print(P_recon_norm)

# Optional: Calculate reprojection error to evaluate the solution
def calculate_reprojection_error(P, world_points, image_points):
    n = world_points.shape[0]
    error = 0
    
    for i in range(n):
        X = world_points[i]
        x = image_points[i]
        
        # Project 3D point to image plane
        x_proj = P @ X
        x_proj = x_proj[:2] / x_proj[2]  # Normalize
        
        # Calculate Euclidean distance
        error += np.sqrt(np.sum((x - x_proj)**2))
    
    return error / n

avg_error = calculate_reprojection_error(P, world_points, image_points)
print(f"\nAverage Reprojection Error: {avg_error:.4f} pixels")

# Visualize the results
plt.figure(figsize=(12, 8))
plt.scatter(image_points[:, 0], image_points[:, 1], color='red', marker='o', s=80, label='Original Points')

# Add point numbers for reference
for i, (x, y) in enumerate(image_points):
    plt.text(x+5, y+5, str(i+1), fontsize=12)

# Project 3D points using the computed projection matrix
projected_points = np.zeros((world_points.shape[0], 2))
for i in range(world_points.shape[0]):
    p = P @ world_points[i]
    projected_points[i] = p[:2] / p[2]

plt.scatter(projected_points[:, 0], projected_points[:, 1], color='blue', marker='x', s=80, label='Projected Points')

# Connect the points to visualize the house structure
# Base (floor)
plt.plot([image_points[0,0], image_points[5,0]], [image_points[0,1], image_points[5,1]], 'k-')
plt.plot([image_points[5,0], image_points[9,0]], [image_points[5,1], image_points[9,1]], 'k-')

# Left wall
plt.plot([image_points[0,0], image_points[8,0]], [image_points[0,1], image_points[8,1]], 'k-')
plt.plot([image_points[8,0], image_points[6,0]], [image_points[8,1], image_points[6,1]], 'k-')

# Right wall
plt.plot([image_points[9,0], image_points[7,0]], [image_points[9,1], image_points[7,1]], 'k-')
plt.plot([image_points[7,0], image_points[4,0]], [image_points[7,1], image_points[4,1]], 'k-')

# Roof
plt.plot([image_points[6,0], image_points[3,0]], [image_points[6,1], image_points[3,1]], 'k-')
plt.plot([image_points[3,0], image_points[4,0]], [image_points[3,1], image_points[4,1]], 'k-')

# Door
plt.plot([image_points[1,0], image_points[2,0]], [image_points[1,1], image_points[2,1]], 'k-')

plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates (origin at top-left)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Camera Calibration: Original vs Projected Points')
plt.legend()
plt.grid(True)
plt.show()