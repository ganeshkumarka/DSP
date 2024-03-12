import cv2
import numpy as np

def laplacian_filter(image):
    laplacian_mask = np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]])
    filtered_image = cv2.filter2D(image, -1, laplacian_mask)
    return filtered_image

def sharpen_image(original_image, filtered_image, alpha=1.0):
    sharpened_image = cv2.addWeighted(original_image, 1 + alpha, filtered_image, -alpha, 0)
    return sharpened_image

if __name__ == "__main__":
    # Read the image
    image = cv2.imread("pic.jpg")

    # Check if the image is loaded successfully
    if image is None:
        print("Error: Image not found.")
    else:
        # Apply Laplacian filter
        filtered_image = laplacian_filter(image)

        # Blend with the original image to sharpen
        sharpened_image = sharpen_image(image, filtered_image)

        # Display the filtered and sharpened images
        cv2.imshow("Filtered Image", filtered_image)
        cv2.imshow("Sharpened Image", sharpened_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the filtered and sharpened images
        cv2.imwrite("filtered_image.jpg", filtered_image)
        cv2.imwrite("sharpened_image.jpg", sharpened_image)
