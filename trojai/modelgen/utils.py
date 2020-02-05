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
