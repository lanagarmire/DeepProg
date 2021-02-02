.. DeepProg documentation master file, created by
   sphinx-quickstart on Fri Dec  6 13:53:29 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DeepProg's documentation!
====================================

.. toctree::
   :maxdepth: 2

   installation.md
   usage.md
   usage_ensemble.md
   usage_advanced.md
   case_study.md
   usage_tuning.md
   LICENSE.rst
   ./api/simdeep.rst


Introduction
------------

This package allows to combine multi-omics data for individual samples together with survival. Using autoencoders (default) or any alternative embedding methods, the pipeline creates new set of features and identifies those linked with survival. In a second time, the samples are clustered with different possible strategies to obtain robust subtypes linked to survival. The robustness of the obtained subtypes can then be tested externally on one or multiple validation datasets and/or the *out-of-bags* samples.  The omic data used in the original study are RNA-Seq, MiR and Methylation. However, this approach can be extended to any combination of omic data. The current package contains the omic data used in the study and a copy of the model computed. However, it is easy to recreate a new model from scratch using any combination of omic data.
The omic data and the survival files should be in tsv (Tabular Separated Values) format and examples are provided. The deep-learning framework to produce the autoencoders uses Keras with either Theano / tensorflow/ CNTK as background.

Access
------

The package is accessible at this link: https://github.com/lanagarmire/DeepProg.

Contribute
----------

- Issue Tracker: github.com/lanagarmire/DeepProg/issues
- Source Code: github.com/lanagarmire/DeepProg

Support
-------

If you are having issues, please let us know.
You can reach us using the following email address:
Olivier Poirion, Ph.D.
o.poirion@gmail.com

Citation
--------

This package refers to our preprint paper: [Multi-omics-based pan-cancer prognosis prediction using an ensemble of deep-learning and machine-learning models](https://www.medrxiv.org/content/10.1101/19010082v1)

License
-------

The project is licensed under the MIT license.
