import torch
from PIL import Image

from src.features.transforms import get_eval_transforms


def test_eval_transforms_output_shape():
    img = Image.new("RGB", (320, 240), color=(100, 150, 200))
    eval_tf = get_eval_transforms(image_size=224, resize_size=256)
    tensor = eval_tf(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_eval_transforms_deterministic():
    img = Image.new("RGB", (300, 300), color=(200, 100, 50))
    eval_tf = get_eval_transforms(image_size=224, resize_size=256)

    t1 = eval_tf(img)
    t2 = eval_tf(img)
    assert torch.equal(t1, t2), "Evaluation transform must be completely deterministic."
