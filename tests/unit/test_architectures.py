import torch
from src.models.architectures import create_resnet50_model, create_mobilenet_v3_model, set_trainable_layers


def test_resnet50_forward_shape():
    model = create_resnet50_model(num_classes=3, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    assert out.shape == (2, 3)


def test_mobilenet_v3_forward_shape():
    model = create_mobilenet_v3_model(num_classes=3, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    assert out.shape == (2, 3)


def test_layer_freezing_phases():
    model = create_resnet50_model(num_classes=3, pretrained=False)

    # Phase 1: Only fc trainable
    t_p1 = set_trainable_layers(model, "resnet50", phase=1)
    for name, param in model.named_parameters():
        if "fc" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False

    # Phase 2: layer4 + fc trainable
    t_p2 = set_trainable_layers(model, "resnet50", phase=2)
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False
