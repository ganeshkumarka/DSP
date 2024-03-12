import cv2
import numpy as np

def display_image(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(img, path):
    cv2.imwrite(path, img)
    print("Image saved successfully.")

if __name__ == "__main__":
    # Read the image
    img = cv2.imread("mammootty.jpg")

    # Check if image is successfully loaded
    if img is None:
        print("Error: Image not found.")
    else:
        # Display the original image
        display_image(img)

        # Add noise to the image
        height, width, _ = img.shape
        mean = 0
        stddev = 25
        for i in range(height):
            for j in range(width):
                for k in range(3):  # Iterate over each channel (RGB)
                    noise = np.random.normal(mean, stddev)
                    img[i, j, k] = np.clip(img[i, j, k] + noise, 0, 255)

        # Display the image with noise
        display_image(img)

        # Save the image with noise
        save_image(img, "noisy_image.jpg")
