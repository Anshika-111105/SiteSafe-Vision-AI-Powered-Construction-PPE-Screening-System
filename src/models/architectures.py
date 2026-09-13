from torch import nn
from torchvision import models


def create_resnet50_model(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """
    Constructs a ResNet50 classifier with an adapted final classification head.
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features=in_features, out_features=num_classes),
    )
    return model


def create_mobilenet_v3_model(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """
    Constructs a MobileNetV3-Large classifier with an adapted final classification head.
    """
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features=in_features, out_features=num_classes)
    return model


def set_trainable_layers(model: nn.Module, model_type: str, phase: int = 1) -> list[str]:
    """
    Configures trainable parameters for Phase 1 (feature extraction) vs Phase 2 (fine-tuning).
    Returns the list of trainable parameter layer names.
    """
    trainable_names = []

    if model_type == "resnet50":
        if phase == 1:
            # Freeze entire backbone, train only the fc classification head
            for name, param in model.named_parameters():
                if "fc" in name:
                    param.requires_grad = True
                    trainable_names.append(name)
                else:
                    param.requires_grad = False
        else:
            # Phase 2: Fine-tune layer4 + fc
            for name, param in model.named_parameters():
                if "layer4" in name or "fc" in name:
                    param.requires_grad = True
                    trainable_names.append(name)
                else:
                    param.requires_grad = False

    elif model_type == "mobilenet_v3_large":
        if phase == 1:
            # Freeze features, train only classifier
            for name, param in model.named_parameters():
                if "classifier" in name:
                    param.requires_grad = True
                    trainable_names.append(name)
                else:
                    param.requires_grad = False
        else:
            # Phase 2: Fine-tune top features (features.15, features.16) + classifier
            for name, param in model.named_parameters():
                if "features.15" in name or "features.16" in name or "classifier" in name:
                    param.requires_grad = True
                    trainable_names.append(name)
                else:
                    param.requires_grad = False
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return trainable_names
