""" Defines valid devices on which models can be trained """
VALID_DEVICES = ['cpu', 'cuda']

""" Defines valid loss functions which can be specified when configuring an optimizer implementing the 
OptimizerInterface"""
VALID_LOSS_FUNCTIONS = ['cross_entropy_loss', 'BCEWithLogitsLoss']

""" Defines valid optimization algorithms which can be specified when configuring an optimizer implementing the 
OptimizerInterface """
VALID_OPTIMIZERS = ['adam', 'sgd']

""" Defines the valid types of data that the modelgen pipeline can handle
"""
VALID_DATA_TYPES = ['image', 'text', 'custom']

MAX_EPOCHS = int(1e6)
