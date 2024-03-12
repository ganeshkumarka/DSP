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
    elif noise_type == 'averaging':
        row, col, ch = image.shape
        noise = np.random.randint(-20, 20, (row, col, ch))  # Random noise between -20 to 20
        noisy = image + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    else:
        raise ValueError("Unsupported noise type. Choose 'gaussian' or 'averaging'.")

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def apply_filter(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))

def main():
    # Load image
    image = cv2.imread("elephant.jpg")
    noisy_images = {}
    filtered_images = {}
    psnr_values = {}

    kernel_sizes = [3, 5, 7, 9, 11, 13, 15]

    # Add noise and apply filters
    for kernel_size in kernel_sizes:
        for noise_type in ['gaussian', 'averaging']:  # Removed 'salt_and_pepper'
            noisy_image = add_noise(image, noise_type)
            noisy_images[(noise_type, kernel_size)] = noisy_image
            filtered_image = apply_filter(noisy_image, kernel_size)
            filtered_images[(noise_type, kernel_size)] = filtered_image
            psnr_value = psnr(image, filtered_image)
            psnr_values[(noise_type, kernel_size)] = psnr_value

            # Save noisy and filtered images
            cv2.imwrite(f"noisy_{noise_type}_{kernel_size}.jpg", noisy_image)
            cv2.imwrite(f"filtered_{noise_type}_{kernel_size}.jpg", filtered_image)

    # Print PSNR values
    for key, value in psnr_values.items():
        print(f"PSNR for {key}: {value}")

if __name__ == "__main__":
    main()

