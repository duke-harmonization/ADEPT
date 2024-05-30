# ADEPT: Adaptive Discretization for Event PredicTion

This repository contains the code for the [ADEPT method found in this manuscript.](https://proceedings.mlr.press/v238/hickey24a/hickey24a.pdf). 

* `adept.py` has the code defining the ADEPT model.
* `example.ipynb` gives an example of how to use the `ADEPT` class.
* `evaluation.py` provides evaluation metrics.
* `generate_bimodal_data.py` generates simulated data with two modes. See Section 4.1 of the manuscript for more information.
* `generate_multimodal_sim_data.py` generates simulated data with four modes. See Section 4.2 of the manuscript for more information.
* `generate_gbsg_data.py` splits the [German Breast Cancer Study Group 2](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.datasets.load_gbsg2.html) data set for training, testing, and validation. See Section 5 of the manuscript for more information.
* `generate_flchain_data.py` splits the [assay of free light chain](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.datasets.load_flchain.html) data set for training, testing, and validation. See Section 5 of the manuscript for more information.
* `pipeline.py` is used to generate the results shown in the Sections 4 and 5 of the manuscript.
* `plotting.py` contains code to make lots for `example.ipynb`

More information about the Stroke data set used in Section 5 of the manuscript [can be found here](https://jamanetwork.com/journals/jama/fullarticle/2800662).

To cite this work use the following citation:

```
@InProceedings{pmlr-v238-hickey24a,
  title = 	 { Adaptive Discretization for Event PredicTion {(ADEPT)} },
  author =       {Hickey, Jimmy and Henao, Ricardo and Wojdyla, Daniel and Pencina, Michael and Engelhard, Matthew},
  booktitle = 	 {Proceedings of The 27th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {1351--1359},
  year = 	 {2024},
  editor = 	 {Dasgupta, Sanjoy and Mandt, Stephan and Li, Yingzhen},
  volume = 	 {238},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {02--04 May},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v238/hickey24a/hickey24a.pdf},
  url = 	 {https://proceedings.mlr.press/v238/hickey24a.html}
}
```
