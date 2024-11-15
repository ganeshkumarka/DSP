import cv2
import os
import numpy as np
from math import log10, sqrt
from skimage.metrics import peak_signal_noise_ratio


def calculate_psnr(original_image_path, compressed_image_path):
    # Read the images
    original_image = cv2.imread(original_image_path)
    compressed_image = cv2.imread(compressed_image_path)

    # Convert images to float32 for PSNR calculation
    original_image = original_image.astype(float)
    compressed_image = compressed_image.astype(float)

    # Ensure images have the same shape
    min_shape = np.min([original_image.shape, compressed_image.shape], axis=0)
    original_image = original_image[:min_shape[0], :min_shape[1], :]
    compressed_image = compressed_image[:min_shape[0], :min_shape[1], :]

    # Calculate the PSNR using scikit-image
    psnr_value = peak_signal_noise_ratio(original_image, compressed_image, data_range=255)

    return psnr_value

# Specify the folder containing the images
folder_path = "img/MBTC"
if __name__ == "__main__":
    # Iterate over intensity values (0.1 to 0.9)
    for intensity in range(1, 10):
        intensity_value = intensity / 10.0
        original_image_path = os.path.join(folder_path, f"noisy_image_intensity_{intensity_value:.1f}.bmp")
        compressed_image_path = os.path.join(folder_path, f"compressed_noisy_image_intensity_{intensity_value:.1f}.bmp_reconstructed.bmp")

        # Calculate PSNR
        psnr_value = calculate_psnr(original_image_path, compressed_image_path)

        # Print the result
        # print(f"PSNR for intensity {intensity_value:.1f}: {psnr_value}")
        print(f"PSNR for intensity  {intensity_value:.1f}: {psnr_value:.2f}")

    img1 = "images\synthetic.bmp"
    img2 = "images\compressedMBTC_img.bmp"
    psnr_value = calculate_psnr(img1, img2)
    print(f"PSNR for original image and compressed image:   {psnr_value:.2f}")