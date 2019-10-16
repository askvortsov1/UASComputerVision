import torchvision.models as models

__all__ = ['densenet121', 'densenet169', 'densenet201', 'densenet161', 'inception_v3', 'squeezenet1_0',
           'squeezenet1_1', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def densenet121(args):
    return models.densenet121(pretrained=args.pretrained, num_classes=args.n_classes)


def densenet169(args):
    return models.densenet169(pretrained=args.pretrained, num_classes=args.n_classes)


def densenet201(args):
    return models.densenet201(pretrained=args.pretrained, num_classes=args.n_classes)


def densenet161(args):
    return models.densenet161(pretrained=args.pretrained, num_classes=args.n_classes)


def inception_v3(args):
    return models.inception_v3(pretrained=args.pretrained, num_classes=args.n_classes)


def resnet18(args):
    return models.resnet18(pretrained=args.pretrained, num_classes=args.n_classes)


def resnet34(args):
    return models.resnet34(pretrained=args.pretrained, num_classes=args.n_classes)


def resnet50(args):
    return models.resnet50(pretrained=args.pretrained, num_classes=args.n_classes)


def resnet101(args):
    return models.resnet101(pretrained=args.pretrained, num_classes=args.n_classes)


def resnet152(args):
    return models.resnet152(pretrained=args.pretrained, num_classes=args.n_classes)


def squeezenet1_0(args):
    return models.squeezenet1_0(pretrained=args.pretrained, num_classes=args.n_classes)


def squeezenet1_1(args):
    return models.squeezenet1_1(pretrained=args.pretrained, num_classes=args.n_classes)



