"""
Data augmentation and transforms module.

Defines light, defensible transforms such as flipping and color jitters,
excluding aggressive augmentation to match industrial inspection constraints.
"""

import torchvision.transforms as T

def get_transforms(image_size: int = 256, horizontal_flip: bool = True):
    """
    Return training and testing transform pipelines.

    Training horizontal flips are dynamically enabled based on the category configuration.
    """
    # Light transformations are chosen because industrial objects are fixed in pose/color
    train_list = [
        T.Resize((image_size, image_size)),
    ]

    if horizontal_flip:
        train_list.append(T.RandomHorizontalFlip(p=0.5))

    # Add minor color jitter (slight changes in brightness and contrast)
    train_list.append(T.ColorJitter(brightness=0.1, contrast=0.1))
    train_list.append(T.ToTensor())
    train_list.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    train_transform = T.Compose(train_list)

    test_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform
