import cv2
import numpy as np
import matplotlib.pyplot as plt
def load_image(path):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
        else:
            print("Image not found")
            return None
    except Exception as e:
        print("Error in loading image")
        print(e)
        return None
def compute_energy(image):
    return np.linalg.norm(img, 2, keepdims=False)
def display_image(img):
    if img is not None:
        plt.imshow(img, cmap="gray")
        plt.show()
    else:
        print("Image not found")
def computeFFT(img):
    return np.fft.fft2(img, norm="ortho")
def compute_iFFT(transformed_img):
    return np.fft.ifft2(transformed_img, norm="ortho")

img_path = "elephant.jpg"
img = load_image(img_path)
imgEnergy = compute_energy(img)
print("The energy of the original image is : ", imgEnergy)

transformed_img = computeFFT(img)
trans_imgEnergy = compute_energy(transformed_img)
print("The energy of the Transformed image is : ", trans_imgEnergy)
absTransformedImg = np.log(np.absolute(transformed_img))

inverse_transformed_img = compute_iFFT(transformed_img)
inverse_imgEnergy = compute_energy(inverse_transformed_img)
print("The energy of the inverse transformed image is:", inverse_imgEnergy)

print(
    "The difference between Original image and Transformed image = ", imgEnergy -  trans_imgEnergy
)

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.xlabel(f'Energy: {round(imgEnergy, 3)}')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(np.log(np.abs(transformed_img) + 1), cmap='gray')
plt.xlabel(f'Energy: {round(trans_imgEnergy, 3)}')
plt.title('DFT of Image')

plt.subplot(1, 3, 3)
plt.imshow(np.abs(inverse_transformed_img), cmap='gray')
plt.xlabel(f'Energy: {round(inverse_imgEnergy, 3)}')
plt.title('Inverse Transformed Image')

plt.suptitle("PARSVEL'S ENERGY THEOREM")
plt.show()

magnitude_spectrum = np.abs(transformed_img)
phase_spectrum = np.angle(transformed_img)

plt.subplot(1, 2, 1)
plt.imshow(np.log(magnitude_spectrum + 1), cmap='gray')
plt.title('Magnitude Spectrum')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(phase_spectrum, cmap='gray')
plt.title('Phase Spectrum')
plt.colorbar()
plt.show()