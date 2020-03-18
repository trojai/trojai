.. TrojAI documentation master file, created by
   sphinx-quickstart on Mon Feb 17 08:38:07 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TrojAI's documentation!
==================================

|

.. image:: images/TrojAI_logo.png
    :width: 49 %
.. image:: images/apl2.png
    :width: 49 %

|
|

.. currentmodule:: trojai

``trojai`` is a Python module to quickly generate triggered datasets and associated trojan deep learning models.  It contains two submodules: ``trojai.datagen`` and ``trojai.modelgen``. ``trojai.datagen`` contains the necessary API functions to quickly generate synthetic data that could be used for training machine learning models. The ``trojai.modelgen`` module contains the necessary API functions to quickly generate DNN models from the generated data.

Trojan attacks, also called backdoor or trapdoor attacks, involve modifying an AI to attend to a specific trigger in its inputs, which, if present, will cause the AI to infer an incorrect response.  For more information, read the :doc:`intro` and our article on `arXiv <https://arxiv.org/abs/2003.07233>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   intro
   installation
   gettingstarted
   contributing
   ack

.. toctree::
   :maxdepth: 3
   :caption: Class reference

   trojai
