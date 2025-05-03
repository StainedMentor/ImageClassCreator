from torchvision import transforms
from torchvision.datasets import ImageFolder
import random
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3),
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.1, 1), ratio=(0.5, 2)),
    transforms.ToTensor()
])

val_test_transform = transforms.Compose([
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor()
])
def load_preprocessed(path, transform):
    dataset = ImageFolder(path,transform = transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return loader


def get_loaders():
    train_loader = load_preprocessed("train", train_transform)
    test_loader = load_preprocessed("test", val_test_transform)
    val_loader = load_preprocessed("val", val_test_transform)
    return train_loader,test_loader,val_loader




if __name__ == "__main__":
    train_loader, test_loader, val_loader = get_loaders()

    # Specify which loader to show HERE:
    images, labels = next(iter(train_loader))
    num_samples = 7
    n_columns = 7

    sample_indices = random.sample(range(len(images)), num_samples)
    samples = [images[i] for i in sample_indices]

    grid = make_grid(samples, nrow=n_columns)
    plt.figure(figsize=(15, 3))
    plt.imshow(TF.to_pil_image(grid))
    plt.axis("off")
    plt.title("Samples")
    plt.show()


