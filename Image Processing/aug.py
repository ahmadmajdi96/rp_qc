import cv2
import numpy as np
import os
import random

def adjust_exposure(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(image, table)

def augment_image(image):
    gamma = random.uniform(0.5, 1.5)
    image = adjust_exposure(image, gamma)

    brightness_factor = random.uniform(0.9, 1.1)
    image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

    if random.choice([True, False]):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    rotations = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]
    rotation = random.choice(rotations)
    image = cv2.rotate(image, rotation)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_shift = random.randint(-10, 10)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return image

def augment_dataset(input_folder, output_folder, num_augmented_images_per_original):
    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)

            for i in range(num_augmented_images_per_original):
                augmented_image = augment_image(image)
                output_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_aug_{i+1}.jpg")
                cv2.imwrite(output_path, augmented_image)

if __name__ == "__main__":
    for dir in os.listdir("raw"):
        print(f"Augmenting at {dir}")
        input_folder = f"raw/{dir}"
        output_folder = f"Dataset/{dir}"
        num_augmented_images_per_original = 100
        augment_dataset(input_folder, output_folder, num_augmented_images_per_original)
