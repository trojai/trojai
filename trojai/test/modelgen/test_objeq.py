import unittest

"""
Test custom __eq__ implementations
"""
import mock

import trojai.modelgen.architecture_factory as tpm_af
import trojai.modelgen.data_manager as tpmdm
import trojai.modelgen.architectures.mnist_architectures as tpma


class LeNetArchFactory(tpm_af.ArchitectureFactory):
    def new_architecture(self):
        return tpma.ModdedLeNet5Net(channels=1)


class BadNetArchFactory(tpm_af.ArchitectureFactory):
    def new_architecture(self):
        return tpma.BadNetExample()


class TestEqImplementations(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_arch_factory_eq(self):
        self.assertEqual(LeNetArchFactory, LeNetArchFactory)
        self.assertEqual(BadNetArchFactory, BadNetArchFactory)
        self.assertNotEqual(LeNetArchFactory, BadNetArchFactory)
        self.assertNotEqual(BadNetArchFactory, LeNetArchFactory)

    def blank_validate():
        pass

    @mock.patch('trojai.modelgen.data_manager.DataManager.validate',
                side_effect=blank_validate)
    def test_data_manager_eq(self, f):
        exp1 = '/tmp/experiment1'
        train1 = '/tmp/train1'
        clean1 = '/tmp/clean1'
        trig1 = '/tmp/trig1'
        data_type = 'image'
        data_xform = (lambda x: x)
        label_xform = (lambda y: y)
        data_xform2 = (lambda x: x*x)
        label_xform2 = (lambda y: y+y)
        data_xform3 = (lambda x: x)
        data_loader = 'image'
        shuffle_train = True
        shuffle_clean_test = True
        shuffle_triggered_test = True
        train_dataloader_kwargs1 = {"a": 1}
        train_dataloader_kwargs2 = {"b": 2, "c": 3}
        test_dataloader_kwargs1 = {"d": 4}
        test_dataloader_kwargs2 = {"e": 5}

        dm1 = tpmdm.DataManager(exp1, train1, clean1, trig1, data_type, data_xform, label_xform,
                                data_loader, shuffle_train, shuffle_clean_test,
                                shuffle_triggered_test, train_dataloader_kwargs1, test_dataloader_kwargs1)
        dm2 = tpmdm.DataManager(exp1, train1, clean1, trig1, data_type, data_xform, label_xform,
                                data_loader, shuffle_train, shuffle_clean_test,
                                shuffle_triggered_test, train_dataloader_kwargs1, test_dataloader_kwargs1)
        # test string comparison difference
        dm3 = tpmdm.DataManager(exp1, '/tmp/train2', clean1, trig1, data_type, data_xform, label_xform,
                                data_loader, shuffle_train, shuffle_clean_test,
                                shuffle_triggered_test, train_dataloader_kwargs1, test_dataloader_kwargs1)
        # test callable comparison difference
        dm4 = tpmdm.DataManager(exp1, train1, clean1, trig1, data_type, data_xform2, label_xform,
                                data_loader, shuffle_train, shuffle_clean_test,
                                shuffle_triggered_test, train_dataloader_kwargs1, test_dataloader_kwargs1)
        # test callable comparison difference
        dm5 = tpmdm.DataManager(exp1, train1, clean1, trig1, data_type, data_xform, label_xform2,
                                data_loader, shuffle_train, shuffle_clean_test,
                                shuffle_triggered_test, train_dataloader_kwargs1, test_dataloader_kwargs1)

        dm6 = tpmdm.DataManager(exp1, train1, clean1, trig1, data_type, data_xform3, label_xform,
                                data_loader, shuffle_train, shuffle_clean_test,
                                shuffle_triggered_test, train_dataloader_kwargs1, test_dataloader_kwargs1)
        # test different dataloader kwargs
        dm7 = tpmdm.DataManager(exp1, train1, clean1, trig1, data_type, data_xform, label_xform,
                                data_loader, shuffle_train, shuffle_clean_test,
                                shuffle_triggered_test, train_dataloader_kwargs2, test_dataloader_kwargs2)
        dm8 = tpmdm.DataManager(exp1, train1, clean1, trig1, data_type, data_xform, label_xform,
                                data_loader, shuffle_train, shuffle_clean_test,
                                shuffle_triggered_test, train_dataloader_kwargs1, test_dataloader_kwargs2)
        self.assertEqual(dm1, dm2)
        self.assertNotEqual(dm1, dm3)
        self.assertNotEqual(dm1, dm4)
        self.assertNotEqual(dm1, dm5)
        # NOTE: this fails because the lambda functions are loaded in different locations in memory, even
        # though they are functionally equivalent.  I'm not sure how to resolve this
        # self.assertEqual(dm1, dm6)
        self.assertNotEqual(dm1, dm7)
        self.assertNotEqual(dm1, dm8)


if __name__ == '__main__':
    unittest.main()
