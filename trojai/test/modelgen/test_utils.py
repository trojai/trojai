import unittest

import torch
import trojai.modelgen.architectures.mnist_architectures as tpma
import os


class MyTestCase(unittest.TestCase):
    def test_load_save(self):
        model_fpath = os.path.join(os.path.dirname(__file__), 'data', 'BadNets_0.2_poison.pt.1')

        model_info = torch.load(model_fpath)

        untrained_model = tpma.BadNetExample()
        loaded_model = tpma.BadNetExample()
        loaded_model.load_state_dict(model_info['state_dict'])

        for param_tensor in untrained_model.state_dict():
            # ensure that the untrained model's weights are not equal to the trained models
            trained_model_weights = model_info['state_dict'][param_tensor]
            untrained_model_weights = untrained_model.state_dict()[param_tensor]
            loaded_model_weights = loaded_model.state_dict()[param_tensor]
            self.assertEqual(torch.all(torch.eq(trained_model_weights, untrained_model_weights)).item(), 0)
            self.assertEqual(torch.all(torch.eq(trained_model_weights, loaded_model_weights)).item(), 1)


if __name__ == '__main__':
    unittest.main()
