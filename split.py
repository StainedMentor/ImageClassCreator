import os
import shutil

from torch.utils.data import random_split
from torchvision.datasets import ImageFolder


def split_data(path):
    full_dataset = ImageFolder(path)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    train_indices, val_indices, test_indices = random_split(
        full_dataset,
        [train_size, val_size, test_size]
    )

    def copy_subset(subset, target_dir):
        os.makedirs(target_dir, exist_ok=True)

        for idx in subset.indices:
            src_path, class_idx = full_dataset.samples[idx]
            class_name = full_dataset.classes[class_idx]
            dest_class_dir = os.path.join(target_dir, class_name)
            os.makedirs(dest_class_dir, exist_ok=True)

            dest_path = os.path.join(dest_class_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dest_path)

    copy_subset(train_indices, "train")
    copy_subset(test_indices, "test")
    copy_subset(val_indices, "val")


split_data("combined")