import cv2

def display_image(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(img, path):
    cv2.imwrite(path, img)
    print("Image saved successfully.")

if __name__ == "__main__":
    print("hello")
    # Read the image
    img = cv2.imread("mammootty.jpg")

    # Check if image is successfully loaded
    if img is None:
        print("Error: Image not found.")
    else:
        # Display the image
        display_image(img)

        # Save the image
        save_image(img, "saved_image.jpg")
