import cv2
import numpy as np
# from matplotlib import pyplot as plt

image = cv2.imread("./dog.png")

mean = 0
std_dev = 20
noise = np.random.normal(mean, std_dev, image.shape)
noisy_image = image + noise.astype(np.uint8)

# plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title("Original")
# plt.subplot(132), plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)), plt.title("Noisy")
# plt.show()

kernel_size = 10
average_filter = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
# smoothed_image = cv2.GaussianBlur(noisy_image, (kernel_size, kernel_size), 0)
smoothed_image = cv2.filter2D(noisy_image, -1, average_filter)

#plt.subplot(131), plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)), plt.title("Noisy")
#plt.subplot(132), plt.imshow(cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2RGB)), plt.title("Smoothed")
#plt.show()

cv2.imshow("Smoothed image",smoothed_image)

mse = np.mean((image - smoothed_image) ** 2)
max_pixel = 255.0
psnr = 10 * np.log10((max_pixel ** 2) / mse)
print("Filter Size : ",kernel_size,"*",kernel_size)
print(f"PSNR: {psnr} dB")

cv2.waitKey(0)
cv2.destroyAllWindows()

