import albumentations as A
import numpy as np


def augmentation(x, y, copies):
    augmented_data_x = []
    augmented_labels_y = np.full((0, 2), 0)

    transform = A.Compose([
        A.CLAHE(),
        A.RandomRotate90(),
        A.Transpose(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=45, p=.65),
        A.Blur(blur_limit=4),
        A.OpticalDistortion(),
        A.GridDistortion(),
        A.HueSaturationValue(),
        A.RandomBrightnessContrast(p=0.5),
    ])

    for img, label in zip(x, y):
        new_y = np.full((copies + 1, 2), label)
        augmented_labels_y = np.concatenate((augmented_labels_y, new_y))
        img *= 255
        img = img.astype(np.uint8)
        augmented_data_x.append(img)
        for i in range(copies):
            augmented_image = transform(image=img)['image']
            augmented_data_x.append(augmented_image)

    augmented_data_x = np.asarray(augmented_data_x)
    augmented_labels_y = np.asarray(augmented_labels_y)

    return augmented_data_x, augmented_labels_y
