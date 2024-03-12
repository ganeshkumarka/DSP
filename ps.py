import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

# Read the image
image = cv2.imread('mammootty.jpg', cv2.IMREAD_GRAYSCALE)

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
psnr_noisy = peak_signal_noise_ratio(image, noisy_image, data_range=255)
psnr_filtered = peak_signal_noise_ratio(image, filtered_image, data_range=255)

# Print PSNR values
print(f'PSNR of Noisy Image: {psnr_noisy:.2f} dB')
print(f'PSNR of Filtered Image: {psnr_filtered:.2f} dB')

# Display the original, noisy, and filtered images
cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image', noisy_image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
