import sys
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_filters import canny, difference_of_gaussians, ex_difference_of_gaussians

def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("L")
    return TF.to_tensor(img).unsqueeze(0) 


def show(original, results):
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(5 * (len(results) + 1), 5))
    for ax, (title, img) in zip(axes, {"Original": original, **results}.items()):
        ax.imshow(img.squeeze().cpu().numpy(), cmap="gray")
        ax.set_title(title, fontsize=40)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = load_image(path).to(device)

    dog = difference_of_gaussians(image.clone(), sigma=1.4, k=1.6)
    dog = (dog - dog.min()) / (dog.max() - dog.min() + 1e-8)

    results = {
        "Canny":        canny(image.clone(), device=device),
        "DoG":          dog,
        "Extended DoG": ex_difference_of_gaussians(image.clone()).float(),
    }

    show(image, results)