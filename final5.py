import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import math
from scipy import signal

def display_image(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(img, path):
    cv2.imwrite(path, img)
    print(f"Image saved successfully at {path}")

def add_gaussian_noise(img, sigma=25):
    # Create a Gaussian noise array
    noise = np.random.normal(0, sigma, img.shape)
    # Add the noise to the original image
    noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return noisy_img

def apply_gaussian_filter(img, sigma=5):
    # Create a Gaussian kernel
    N = 6 * sigma
    t = np.linspace(-3 * sigma, 3 * sigma, N)
    gau = (1 / (math.sqrt(2 * math.pi) * sigma)) * np.exp(-0.5 * (t / sigma) ** 2)
    kernel = gau[:, np.newaxis] * gau[np.newaxis, :]
    # Apply the Gaussian filter
    filtered_img = signal.fftconvolve(img, kernel[:, :, np.newaxis], mode='same')
    # Rescale to [0, 255]
    filtered_img = (filtered_img - filtered_img.min()) / (filtered_img.max() - filtered_img.min()) * 255
    return filtered_img.astype(np.uint8)

if __name__ == "__main__":
    # Read the original image
    original_img = cv2.imread("elephant.jpg")

    # Check if image is successfully loaded
    if original_img is None:
        print("Error: Image not found.")
    else:
        # Add Gaussian noise to the original image
        noisy_img = add_gaussian_noise(original_img)
        display_image(noisy_img)
        save_image(noisy_img, "noisy_image.jpg")

        # Apply Gaussian filter to the noisy image
        for sigma in range(3, 16, 2): # Start from 3, end at 15, step by 2
            filtered_image = apply_gaussian_filter(noisy_img, sigma)
            display_image(filtered_image)
            save_path = f"filtered_image_sigma_{sigma}.jpg"
            save_image(filtered_image, save_path)

            # Calculate PSNR values
            psnr_noisy_filtered = peak_signal_noise_ratio(noisy_img, filtered_image, data_range=255)
            psnr_original_noisy = peak_signal_noise_ratio(original_img, noisy_img, data_range=255)
            psnr_original_filtered = peak_signal_noise_ratio(original_img, filtered_image, data_range=255)

            # Print PSNR values
            print(f'Sigma: {sigma}')
            print(f'PSNR between Noisy and Filtered Image: {psnr_noisy_filtered:.2f} dB')
            print(f'PSNR between Original and Noisy Image: {psnr_original_noisy:.2f} dB')
            print(f'PSNR between Original and Filtered Image: {psnr_original_filtered:.2f} dB')
            print("--------------------------------------------------")
