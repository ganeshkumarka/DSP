import cv2
import numpy as np

def add_noise(image, noise_type='gaussian'):
    if noise_type == 'gaussian':
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)
    else:
        raise ValueError("Unsupported noise type. Choose 'gaussian'.")

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def apply_filter(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def main():
    # Load image
    image = cv2.imread("mammootty.jpg")
    noisy_images = {}
    filtered_images = {}
    psnr_values = {}

    kernel_sizes = [3, 5, 7, 9, 11, 13, 15]

    # Add noise and apply filters
    for kernel_size in kernel_sizes:
        noisy_image = add_noise(image)
        noisy_images[kernel_size] = noisy_image
        filtered_image = apply_filter(noisy_image, kernel_size)
        filtered_images[kernel_size] = filtered_image
        psnr_value = psnr(image, filtered_image)
        psnr_values[kernel_size] = psnr_value

        # Save noisy and filtered images
        cv2.imwrite(f"noisy_gaussian_{kernel_size}.jpg", noisy_image)
        cv2.imwrite(f"filtered_gaussian_{kernel_size}.jpg", filtered_image)

    # Print PSNR values
    for key, value in psnr_values.items():
        print(f"PSNR for kernel size {key}: {value}")

if __name__ == "__main__":
    main()
