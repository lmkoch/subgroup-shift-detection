# Hidden in Plain Sight: Subgroup Shifts Escape OOD Detection

This is the code used in the MIDL 2022 submission "Hidden in Plain Sight: Subgroup Shifts Escape OOD Detection" (https://openreview.net/pdf?id=aZgiUNye2Cz).

![image](https://drive.google.com/uc?export=view&id=1-uw5xMSCdg8ZxIBM3UOV5y9n3g7VMlc_)


# Prerequisites

* **Python environment**

  ````
  > pip install -e .
  ````

  Note that for experiments on histopathology data, we use our custom fork of the `wilds` package. You don't need to worry about this if you install from the setup file directly.

* **Data** 

  For both MNIST and Camelyon17, the data is automatically to `data_root` directory specified in config the first time the dataset is created. For Camelyon17, this may take a while.

* **Task classifiers (for ODIN and MUKS)**

  For the individual OOD detection as well as the MUKS hypothesis tests, you need trained task classifiers. These are downloaded automatically to the paths specified in the config file (key `task_classifier_path`).
  
  If you want to train task classifiers yourself, follow the instructions [below](#other)

# Experiments

## Configurations

All experiments are fully specified by config files which can be found in `./config`. Please adjust paths in there as needed.

The provided config templates assume the following coarse project structure:

- `./experiments`: all your experiments go here
- `./data`: all the raw data is here. 

## Run individual configurations

To run an individual configuration, you can do

```
# For population-level subgroup shift detection
python ./scripts/run_main_exp.py --exp_dir EXP_DIR --config_file CONFIG_FILE --eval_splits test

# For individual OOD detection
python ./scripts/run_odin_exp.py --exp_dir EXP_DIR --config_file CONFIG_FILE
```

A folder will be placed inside `exp_dir` named after a unique hash of the config file.
Trained models and results will be placed there.

## Reproduce paper experiments

To reproduce the experiments in the paper, run 

````
python ./scripts/run_all_experiments.py --exp_base_dir ./experiments
````

Adjust the experiment directory as needed. If you have access to a slurm cluster, you can use the options `--slurm True --slurm_email your@email.com [+slurm resource configuration]` to accelerate the compute heavy operations.

Note: Make sure all data and models are downloaded before running parallel experiments, as this will result in conflicts due to simultaneous downloads. You can do this by running

````
python ./scripts/download.py --data_mnist DATA_PATH_MNIST --data_camelyon DATA_PATH_CAMELYON --model_mnist MODEL_PATH_MNIST --model_mnist_no5 MODEL_PATH_MNIST_NO5 --model_camelyon MODEL_PATH_CAMELYON
````

Here. The paths should correspond to the paths specified in the config files.

Once the jobs are completed, you can generate the paper figures with 

```
python ./scripts/plot_results.py
```

Check whether default paths work for you (e.g. with `python plot_results.py --help`), otherwise pass the correct ones.


# Citation

If you use this code, please cite

````
@inproceedings{koch2022subgroup,
  title      = {Hidden in Plain Sight: Subgroup Shifts Escape OOD Detection},
  author     = {Koch, Lisa M and Sch{\"u}rch, Christian M and Gretton, Arthur and Berens, Philipp},
  booktitle  = {Proc. Medical Imaging with Deep Learning (MIDL) - under review},
  year       = {2022},
}
````

Please also note that code segments from related work were used. If use them, please also cite:

````
@inproceedings{liu2020deepkernel,
  title      = {Learning {Deep} {Kernels} for {Non}-{Parametric} {Two}-{Sample} {Tests}},
  author     = {Liu, Feng and Xu, Wenkai and Lu, Jie and Zhang, Guangquan and Gretton, Arthur and Sutherland, Danica J},
  booktitle  = {Proc. International Conference on Machine Learning (ICML)},
  year       = {2020},
}

@inproceedings{liang2018odin,
  title      = {Enhancing {The} {Reliability} of {Out}-of-distribution {Image} {Detection} in {Neural} {Networks}},
  author     = {Liang, Shiyu and Li, Yixuan and Srikant, R.},
  booktitle  = {Proc. International Conference on Learning Representations (ICLR)},
  year       = {2018}
}

````

# Other

## Train task classifiers yourself

If you want to train task classifiers yourself, follow the instructions below. **Note**: Set the correct paths to the classification models in the config and script parameters.

- **MNIST:** Use the training script provided
    ````
    > python ./scripts/train_mnist_classifier.py --config_file CONFIG_FILE
    ````

- **Camelyon17:** We used the `wilds` package for training a Densenet on Camelyon17 data. Since we require different train, validation and test splits, you need to use our [fork](https://github.com/lmkoch/wilds) of the wilds package. Clone it, then run the training script with the following parameters

    ````
    > python examples/run_expt.py --dataset camelyon17 --root_dir /path/to/data \
        --split_scheme vanilla --seed 9 
    ````
