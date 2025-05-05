########################## EXERCISE 2 ##########################
import cv2 
import numpy as np

def main():
    # Read the input image
    img = cv2.imread("homework6.pgm", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not read the image.")
        return
    # Source and destination points definition
    src_points = np.array([[244,263], [238,353], [199,350], [201,262]], dtype=np.float32)
    dst_points = np.array([[232,216], [232,311], [197,311], [197,216]], dtype=np.float32)
    # Compute the homography matrix
    h_matrix, _ = cv2.findHomography(dst_points, src_points)
    # Create an empty output image
    output_size = (300, 370)
    # Use the inverse of h and bilinear interpolation img for computing the intensities of the pixels in B.
    B = cv2.warpPerspective(img, h_matrix, output_size, flags=cv2.INTER_LINEAR)

    # Save the output img
    cv2.imwrite("Images/rectified_output.pgm", B)
    cv2.imshow("Rectified Image", B)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    main()