from trojai.modelgen.utils import resave_trojai_model_as_dict
# TODO: import the class which implements the network that is to be resaved


def resave_models(files):
    """
    Convert PyTorch models saved by simple torch.save(model) calls to TrojAI model specification dictionaries.
    :param files: (list of strings) File names to be converted
    """
    for f_name in files:
        resave_trojai_model_as_dict(f_name)


if __name__ == "__main__":
    file_names = ["model1.pt", "model2.pt"]
    resave_models(file_names)
