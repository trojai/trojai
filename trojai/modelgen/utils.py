import torch
import pkg_resources


def make_trojai_model_dict(model):
    """
    Create a TrojAI approved dictionary specification of a PyTorch model for saving to a file. E.g. for a trained model
        'model':
                        save_dict = make_trojai_model_dict(model)
                        torch.save(save_dict, filename)
    :param model: (torch.nn.Module) The desired model to be saved.
    :return: (dict) dictionary containing TrojAI approved information about the model, which can also be used for
        later loading the model.
    """
    # TODO: implement trojai.__version__ and remove pkg_resources module
    return {"class_name": model.__class__.__name__,
            "trojai_version": pkg_resources.require("trojai")[0].version,
            "state_dict": model.state_dict()}


def resave_trojai_model_as_dict(file, new_loc=None):
    """
    Load a fully serialized Pytorch model (i.e. whole model was saved instead of a specification) and save it as a
        TrojAI style dictionary specification.
    :param file: (str) Location of the file to re-save
    :param new_loc: (str) Where to save the file if replacing the original is not desired
    """
    model = torch.load(file)
    model_dict = make_trojai_model_dict(model)
    if new_loc:
        torch.save(model_dict, new_loc)
    else:
        torch.save(model_dict, file)


def clamp(X, l, u, cuda=True):
    """
    Clamps a tensor to lower bound l and upper bound u.
    :param X: the tensor to clamp.
    :param l: lower bound for the clamp.
    :param u: upper bound for the clamp.
    :param cuda: whether the tensor should be on the gpu.
    """

    if type(l) is not torch.Tensor:
        if cuda:
            l = torch.cuda.FloatTensor(1).fill_(l)
        else:
            l = torch.FloatTensor(1).fill_(l)
    if type(u) is not torch.Tensor:
        if cuda:
            u = torch.cuda.FloatTensor(1).fill_(u)
        else:
            u = torch.FloatTensor(1).fill_(u)
    return torch.max(torch.min(X, u), l)


def get_uniform_delta(shape, eps, requires_grad=True):
    """
    Generates a troch uniform random matrix of shape within +-eps.
    :param shape: the tensor shape to create.
    :param eps: the epsilon bounds 0+-eps for the uniform random tensor.
    :param requires_grad: whether the tensor requires a gradient.
    """
    delta = torch.zeros(shape).cuda()
    delta.uniform_(-eps, eps)
    delta.requires_grad = requires_grad
    return delta