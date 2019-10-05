import torchvision.models as models

__all__ = ['densenet121', 'densenet169', 'densenet201', 'densenet161', 'inception_v3', 'squeezenet1_0',
           'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn','vgg19_bn', 'vgg19']


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


def vgg11(args):
    return models.vgg11(pretrained=args.pretrained, num_classes=args.n_classes)


def vgg11_bn(args):
    return models.vgg11_bn(pretrained=args.pretrained, num_classes=args.n_classes)


def vgg13(args):
    return models.vgg13(pretrained=args.pretrained, num_classes=args.n_classes)


def vgg13_bn(args):
    return models.vgg13_bn(pretrained=args.pretrained, num_classes=args.n_classes)


def vgg16(args):
    return models.vgg16(pretrained=args.pretrained, num_classes=args.n_classes)


def vgg16_bn(args):
    return models.vgg16_bn(pretrained=args.pretrained, num_classes=args.n_classes)


def vgg19(args):
    return models.vgg19(pretrained=args.pretrained, num_classes=args.n_classes)


def vgg19_bn(args):
    return models.vgg19_bn(pretrained=args.pretrained, num_classes=args.n_classes)

