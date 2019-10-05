import torch.optim as optim

__all__ = ['adadelta', 'adagrad', 'adam', 'adamax', 'rmsprop', 'sgd']


def adam(net_params, args):
    return optim.Adam(net_params, lr=args.lr, eps=args.eps, betas=(args.beta1, args.beta2),
                      weight_decay=args.weight_decay, amsgrad=args.amsgrad)


def sgd(net_params, args):
    return optim.SGD(net_params, lr=args.lr, momentum=args.momentum, dampening=args.dampening,
                     weight_decay=args.weight_decay, nesterov=args.nesterov)


def adagrad(net_params, args):
    return optim.Adagrad(net_params, lr=args.lr,
                         lr_decay=args.lr_decay, weight_decay=args.weight_decay)


def adamax(net_params, args):
    return optim.Adamax(net_params, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps,
                        weight_decay=args.weight_decay)


def adadelta(net_params, args):
    return optim.Adadelta(net_params, lr=args.lr, rho=args.rho, eps=args.eps, weight_decay=args.weight_decay)


def rmsprop(net_params, args):
    return optim.RMSprop(net_params, lr=args.lr, alpha=args.alpha, eps=args.eps, weight_decay=args.weight_decay,
                         momentum=args.momentum, centered=args.centered)
