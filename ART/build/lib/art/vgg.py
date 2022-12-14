from functools import partial
from typing import Union, List, Dict, Any, Optional, cast

import torch
import torch.nn as nn

#from ..transforms._presets import ImageClassification
#from ..utils import _log_api_usage_once
#from ._api import WeightsEnum, Weights
#from ._meta import _IMAGENET_CATEGORIES
#from ._utils import handle_legacy_interface, _ovewrite_named_param


__all__ = [
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
]


class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        #_log_api_usage_once(self)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            #nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print(x.shape)
        x = self.features(x)
        #print(x.shape)
        x = self.avgpool(x)
        #print(x.shape)
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = self.classifier(x)
        #print(x.shape)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(cfg: str, batch_norm: bool, weights: Any, progress: bool, **kwargs: Any) -> VGG:
    if weights is not None:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    return model




def vgg11() -> VGG:
    """VGG-11 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    Args:
        weights (:class:`~torchvision.models.VGG11_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG11_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.VGG11_Weights
        :members:
    """
    #weights = VGG11_Weights.verify(weights)

    return _vgg("A", False, None, True)


def vgg11_bn() -> VGG:
    """VGG-11-BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    Args:
        weights (:class:`~torchvision.models.VGG11_BN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG11_BN_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.VGG11_BN_Weights
        :members:
    """
    #weights = VGG11_BN_Weights.verify(weights)

    return _vgg("A", True, None, True)


def vgg13() -> VGG:
    """VGG-13 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    Args:
        weights (:class:`~torchvision.models.VGG13_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG13_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.VGG13_Weights
        :members:
    """
    #weights = VGG13_Weights.verify(weights)

    return _vgg("B", False, None, True)


def vgg13_bn() -> VGG:
    """VGG-13-BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    Args:
        weights (:class:`~torchvision.models.VGG13_BN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG13_BN_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.VGG13_BN_Weights
        :members:
    """
    #weights = VGG13_BN_Weights.verify(weights)

    return _vgg("B", True, None, True)


def vgg16() -> VGG:
    """VGG-16 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    Args:
        weights (:class:`~torchvision.models.VGG16_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG16_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.VGG16_Weights
        :members:
    """
    #weights = VGG16_Weights.verify(weights)

    return _vgg("D", False, None, True)


def vgg16_bn() -> VGG:
    """VGG-16-BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    Args:
        weights (:class:`~torchvision.models.VGG16_BN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG16_BN_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.VGG16_BN_Weights
        :members:
    """
    #weights = VGG16_BN_Weights.verify(weights)

    return _vgg("D", True, None, True)


def vgg19() -> VGG:
    """VGG-19 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    Args:
        weights (:class:`~torchvision.models.VGG19_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG19_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.VGG19_Weights
        :members:
    """
    #weights = VGG19_Weights.verify(weights)

    return _vgg("E", False, None, True)


def vgg19_bn() -> VGG:
    """VGG-19_BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    Args:
        weights (:class:`~torchvision.models.VGG19_BN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG19_BN_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.VGG19_BN_Weights
        :members:
    """
    #weights = VGG19_BN_Weights.verify(weights)

    return _vgg("E", True, None, True)


