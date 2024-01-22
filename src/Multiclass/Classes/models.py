import torch.nn as nn
import torchvision.models as models

class ModelFactory:
    """
    Factory class to create various models for classification tasks.
    """

    @staticmethod
    def make_resnet(pretrained=True, device='cpu'):
        """
        Creates a ResNet model for classification.

        Parameters:
        - pretrained (bool): If True, uses pretrained weights. Default is True.
        - device (str): Device on which to load the model. Default is 'cpu'.

        Returns:
        - model: ResNet model.
        """
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        else:
            model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
        model = model.to(device)
        return model

    @staticmethod
    def make_efficientnet(pretrained=True, device='cpu'):
        """
        Creates an EfficientNet model for classification.

        Parameters:
        - pretrained (bool): If True, uses pretrained weights. Default is True.
        - device (str): Device on which to load the model. Default is 'cpu'.

        Returns:
        - model: EfficientNet model.
        """
        if pretrained:
            weights = models.EfficientNet_B1_Weights.DEFAULT
            model = models.efficientnet_b1(weights=weights)
        else:
            model = models.efficientnet_b1(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 3)
        model = model.to(device)
        return model

    @staticmethod
    def make_densenet(pretrained=True, device='cpu'):
        """
        Creates a DenseNet model for classification.

        Parameters:
        - pretrained (bool): If True, uses pretrained weights. Default is True.
        - device (str): Device on which to load the model. Default is 'cpu'.

        Returns:
        - model: DenseNet model.
        """
        if pretrained:
            weights = models.DenseNet161_Weights.DEFAULT
            model = models.densenet161(weights=weights)
        else:
            model = models.densenet161(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 3)
        model = model.to(device)
        return model

    @staticmethod
    def make_swin(pretrained=True, device='cpu'):
        """
        Creates a Swin Transformer model for classification.

        Parameters:
        - pretrained (bool): If True, uses pretrained weights. Default is True.
        - device (str): Device on which to load the model. Default is 'cpu'.

        Returns:
        - model: Swin Transformer model.
        """
        if pretrained:
            weights = models.Swin_T_Weights.DEFAULT
            model = models.swin_t(weights=weights)
        else:
            model = models.swin_t(weights=None)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(in_features=num_ftrs, out_features=3, bias=True)
        model = model.to(device)
        return model

    @staticmethod
    def make_swin_s(pretrained=True, device='cpu'):
        """
        Creates a smaller Swin Transformer model for classification.

        Parameters:
        - pretrained (bool): If True, uses pretrained weights. Default is True.
        - device (str): Device on which to load the model. Default is 'cpu'.

        Returns:
        - model: Smaller Swin Transformer model.
        """
        if pretrained:
            weights = models.Swin_S_Weights.DEFAULT
            model = models.swin_s(weights=weights)
        else:
            model = models.swin_s(weights=None)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(in_features=num_ftrs, out_features=3, bias=True)
        model = model.to(device)
        return model

    @staticmethod
    def make_swin_v2(pretrained=True, device='cpu'):
        """
        Creates a second version of Swin Transformer model for classification.

        Parameters:
        - pretrained (bool): If True, uses pretrained weights. Default is True.
        - device (str): Device on which to load the model. Default is 'cpu'.

        Returns:
        - model: Swin Transformer V2 model.
        """
        if pretrained:
            weights = models.Swin_V2_S_Weights.DEFAULT
            model = models.swin_v2_s(weights=weights)
        else:
            model = models.swin_v2_s(weights=None)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(in_features=num_ftrs, out_features=3, bias=True)
        model = model.to(device)
        return model

    @staticmethod
    def make_swin_v2_b(pretrained=True, device='cpu'):
        """
        Creates a larger version of Swin Transformer model for classification.

        Parameters:
        - pretrained (bool): If True, uses pretrained weights. Default is True.
        - device (str): Device on which to load the model. Default is 'cpu'.

        Returns:
        - model: Larger Swin Transformer V2 model.
        """
        if pretrained:
            weights = models.Swin_V2_B_Weights.DEFAULT
            model = models.swin_v2_b(weights=weights)
        else:
            model = models.swin_v2_b(weights=None)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(in_features=num_ftrs, out_features=3, bias=True)
        model = model.to(device)
        return model