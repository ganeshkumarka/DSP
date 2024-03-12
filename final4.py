import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

def display_image(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(img, path):
    cv2.imwrite(path, img)
    print(f"Image saved successfully at {path}")

if __name__ == "__main__":
    # Read the original image
    original_img = cv2.imread("elephant.jpg")

    # Check if image is successfully loaded
    if original_img is None:
        print("Error: Image not found.")
    else:
        # Add noise to the original image
        height, width, _ = original_img.shape
        mean = 0
        stddev = 25
        noisy_img = original_img.copy()
        for i in range(height):
            for j in range(width):
                for k in range(3): # Iterate over each channel (RGB)
                    noise = np.random.normal(mean, stddev)
                    noisy_img[i, j, k] = np.clip(original_img[i, j, k] + noise, 0, 255)

        # Display and save the noisy image
        display_image(noisy_img)
        save_image(noisy_img, "noisy_image.jpg")

        # Iterate over kernel sizes from 3x3 to 15x15
        for kernel_size in range(3, 16, 2): # Start from 3, end at 15, step by 2
            # Create a kernel of the current size and apply the filter
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            filtered_image = cv2.filter2D(noisy_img, -1, kernel)

            # Display and save the filtered image
            display_image(filtered_image)
            save_path = f"filtered_image_{kernel_size}x{kernel_size}.jpg"
            save_image(filtered_image, save_path)

            # Calculate PSNR values
            psnr_noisy_filtered = peak_signal_noise_ratio(noisy_img, filtered_image, data_range=255)
            psnr_original_noisy = peak_signal_noise_ratio(original_img, noisy_img, data_range=255)
            psnr_original_filtered = peak_signal_noise_ratio(original_img, filtered_image, data_range=255)

            # Print PSNR values
            print(f'Kernel Size: {kernel_size}x{kernel_size}')
            print(f'PSNR between Noisy and Filtered Image: {psnr_noisy_filtered:.2f} dB')
            print(f'PSNR between Original and Noisy Image: {psnr_original_noisy:.2f} dB')
            print(f'PSNR between Original and Filtered Image: {psnr_original_filtered:.2f} dB')
            print("--------------------------------------------------")
