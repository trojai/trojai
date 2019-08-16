RANDOM_STATE_DRAW_LIMIT = 2**32 - 1
'''
In the data generation process, every new entity that is generated gets a new random seed by drawing from 
np.random.RandomState.randint(), where the RandomState object comes from a master RandomState created at the 
beginning of the data generation process.  The constant RANDOM_STATE_DRAW_LIMIT defines the argument passed into the 
randint(...) call.

The reason we create a new seed for every Entity is to enable reproducibility.  Each Entity that is created may go 
through a series of transformations that include randomness at various stages.  As such, having a seed associated 
with each Entity will enable us to reproduce those specific random variations easily.
'''
