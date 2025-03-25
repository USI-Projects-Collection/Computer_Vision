# Set the stage
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import matplotlib.pyplot as plt

# Read and view an image
A = cv2.imread("dog.pgm", cv2.IMREAD_GRAYSCALE)
L = 256
cv2_imshow(A)

# Get height and width of image
M = A.shape[0]
N = A.shape[1]

# Cumulative distribution function (of the original image)
P, bins, ignored = plt.hist(A.ravel(), L, [0,L], density=True)
S = np.copy(P)
for i in range(1,L):
  S[i] = S[i-1] + S[i]
S = (L-1) * S
plt.plot(range(0,L), S*max(P)/L, color='r')
plt.show()

# Histogram equalization (following the book)
T1 = np.round(S)
B1 = np.zeros((M,N))
for i in range(0,M):
  for j in range(0,N):
    B1[i,j] = T1[A[i,j]]
cv2_imshow(B1)

# Cumulative distribution function (after histogram equalization)
Q1, bins, ignored = plt.hist(B1.ravel(), L, [0,L], density=True)
S1 = np.copy(Q1)
for i in range(1,L):
  S1[i] = S1[i-1] + S1[i]
S1 = (L-1) * S1
plt.plot(range(0,L), S1*max(Q1)/L, color='r')
plt.show()

# Histogram equalization (following Wikipedia)
Smin = np.min(S)
Shat = (S-Smin)/(1-Smin/(L-1))
T2 = np.round(Shat)
B2 = np.zeros((M,N))
for i in range(0,M):
  for j in range(0,N):
    B2[i,j] = T2[A[i,j]]
cv2_imshow(B2)

Q2, bins, ignored = plt.hist(B2.ravel(), L, [0,L], density=True)
S2 = np.copy(Q2)
for i in range(1,L):
  S2[i] = S2[i-1] + S2[i]
S2 = (L-1) * S2
plt.plot(range(0,L), S2*max(Q2)/L, color='r')
plt.show()

print(abs(T1-T2))

# Histogram equalization (following the slides)
Stilde = np.zeros((L))
Stilde[0] = 0.5 * (L*P[0] - 1)
for i in range(1,L):
  Stilde[i] = Stilde[i-1] + 0.5 * L * (P[i-1]+P[i])
T3 = np.round(Stilde)
B3 = np.zeros((M,N))
for i in range(0,M):
  for j in range(0,N):
    B3[i,j] = T3[A[i,j]]
cv2_imshow(B3)

Q3, bins, ignored = plt.hist(B3.ravel(), L, [0,L], density=True)
S3 = np.copy(Q3)
for i in range(1,L):
  S3[i] = S3[i-1] + S3[i]
S3 = (L-1) * S3
plt.plot(range(0,L), S3*max(Q3)/L, color='r')
plt.show()

print(abs(T1-T3))

