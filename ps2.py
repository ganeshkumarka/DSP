import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

# Read the image
image = cv2.imread('elephant.jpg', cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise to the image
mean = 0
stddev = 25
noise = np.random.normal(mean, stddev, image.shape)
noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)

# Define the size of the filter (e.g., 3x3)
filter_size = 3

# Apply averaging filter
filtered_image = cv2.blur(noisy_image, (filter_size, filter_size))

# Calculate PSNR values
mse_noisy = np.mean((image - noisy_image) ** 2)
mse_filtered = np.mean((image - filtered_image) ** 2)
psnr_noisy = 10 * np.log10((255 ** 2) / mse_noisy)
psnr_filtered = 10 * np.log10((255 ** 2) / mse_filtered)

# Print PSNR values
print(f'PSNR of Noisy Image: {psnr_noisy:.2f} dB')
print(f'PSNR of Filtered Image: {psnr_filtered:.2f} dB')

# Display the original, noisy, and filtered images
cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image', noisy_image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
