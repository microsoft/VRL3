# VRL3 codebase
Official code for the paper VRL3: A Data-Driven Framework for Visual Deep Reinforcement Learning. Summary site: https://sites.google.com/nyu.edu/vrl3. 

Code has just been released and the entire codebase is re-written to make it cleaner and improve readability, so it is possible you might run into an problem, in that case please do not hesitate to post an issue.

We are also doing some further clean-up of the code now. This repo will be updated. 

### updates:
<sup>03/30/2023: added example plot function and a quick tutorial.</sup>


## Repo structure and important files: 
```
VRL3 # this repo
│   README.md # read this file first!
└───docker # dockerfile with all dependencies
└───plot_utils # code for plotting, still working on it now...
└───src
    │   train_adroit.py # setup and main training loop
    │   vrl3_agent.py # agent class, code for stage 2, 3
    │   train_stage1.py # code for stage 1 pretraining on imagenet
    │   stage1_models.py # the encoder classes pretrained in stage 1
    └───cfgs_adroit # configuration files with all hyperparameters

# download these folders from the google drive link
vrl3data 
└───demonstrations # adroit demos 
└───trained_models # pretrained stage 1 models
vrl3examplelogs 
└───rrl  # rrl training logs
└───vrl3 # vrl3 with default hyperparams logs
```

To get started, download this repo and download adroit demos, pretrained models, and example logs with the following link: 
https://drive.google.com/drive/folders/14rH_QyigJLDWsacQsrSNV7b0PjXOGWwD?usp=sharing

## Set up environment
The recommended way is to just use the dockerfile I provided and follow the tutorial here. You can also look at the dockerfile to know the exact dependencies or modify it to build a new dockerfile. 

### Setup with docker
If you have a local machine with gpu, or your cluster allows docker (you have sudo), then you can just pull my docker image and run code there. (Newest version is 1.5, where the mujoco slow rendering with gpu issue is fixed). 
```
docker pull docker://cwatcherw/vrl3:1.5
```

Now, `cd` into a directory where you have the `VRL3` folder (this repo), and also the `vrl3data` folder that you downloaded from my google drive link. 
Then, mount `VRL3/src` to `/code`, and mount `vrl3data` to `/vrl3data` (you can also mount to other places, but you will need to adjust some commands or paths in the config files):
```
docker run -it --rm --gpus all -v "$(pwd)"/VRL3/src:/code -v "$(pwd)"/vrl3data:/vrl3data  docker://cwatcherw/vrl3:1.5
```
Now you should be inside the docker container. Refer to the "Run experiments" section now. 

### Run experiments
Once you get into the container (either docker or singularity), first run the following commands so the paths are correct. Very important especially on singularity since it uses automount which can mess up the paths. (newest version code now uses `os.environ` to do these so you can also skip this step.) 
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=/workspace/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL=egl
```

Go to the VRL3 code directory that you mounted.
```
cd /code
```

First quickly check if mujoco is using your GPU correctly for rendering. If everything is correct, you should see the program print out the computation time for 1000 rendering (if it's first time mujoco is imported then there will also be mujoco build messages which takes a few minutes). The time used to do rendering 1000 times should be < 0.5 seconds. 
```
python testing/computation_time_test.py
```


Then you can start run VRL3:
```
cd /code
python train_adroit.py task=door
```

For first-time setup, use `debug=1` to do a quick test run to see if the code is working. This will reduce training epochs and change many other hyperparameters so you will get a full run in a few minutes. 
```
python train_adroit.py task=door debug=1
```

You can also run with different hyperparameters, see the `config.yaml` for a full list of them. For example: 
```
python train_adroit.py task=door stage2_n_update=5000 agent.encoder_lr_scale=0.1
```



### Setup singularity 
If your cluster does not allow sudo (for example, NYU's slurm HPC), then you can use singularity, it is similar to docker. But you might need to modify some of the commands depends on how your cluster is being managed. Here is an example setup on the NYU Greene HPC.

Set up singularity container (this will make a folder called `sing` in your scratch directory, and then build a singularity sandbox container called `vrl3sing`, using the `cwatcherw/vrl3:1.5` docker container which I put on my docker hub):
```
mkdir /scratch/$USER/sing/
cd /scratch/$USER/sing/
singularity build --sandbox vrl3sing docker://cwatcherw/vrl3:1.5
```

For example, on NYU HPC, start interactive session (if your school has a different hpc system, consult your hpc admin): 
```
srun --pty --gres=gpu:1 --cpus-per-task=4 --mem 12000 -t 0-06:00 bash
```
Here, by default VRL3 uses 4 workers for dataloader, so we request 4 cpus. Once the job is allocated to you, go the `sing` folder where you have your container, then run it:
```
cd /scratch/$USER/sing
singularity exec --nv -B /scratch/$USER/sing/VRL3/src:/code -B /scratch/$USER/sing/vrl3sing/opt/conda/lib/python3.8/site-packages/mujoco_py/:/opt/conda/lib/python3.8/site-packages/mujoco_py/ -B /scratch/$USER/sing/vrl3data:/vrl3data /scratch/$USER/sing/vrl3sing bash
```
We mount the `mujoco_py` package folder because singularity files by default are read-only, and the older version of mujoco_py wants to modify files, which can be problematic. (And Adroit env relies on older version of mujoco so we have to deal with it...)

After the singularity container started running, now refer to the "Run experiments" section.

## Plotting example
If you like to use the plotting functions we used, you will need `matplotlib`, `seaborn` and some other basic packages to use the plotting programs. You can also use your own plotting functions. 

An example is given in `plot_utils/vrl3_plot_example.py`. To use it: 
1. make sure you downloaded the `vrl3examplelogs` folder from the drive link and unzipped it. 
2. in `plot_utils/vrl3_plot_example.py`, change the `base_dir` path to where the `vrl3examplelogs` folder is on your computer. 
3. similarlly, change `base_save_dir` path to where you want the figures to be generated. 
4. run `plot_utils/vrl3_plot_example.py`, this will generate a few figures comparing success rate between RRL and VRL3 to the specified path. 

(All ablation experiment logs generated during the VRL3 research are in the folder `vrl3logs` from the drive link. `plot_utils/vrl3_plot_runner.py` was used to generate figures in the paper. Still need further clean up.)

## Some hyperparameter details
- BC loss: in the config files, I now by default disable all BC loss since our ablations show they are not really helping. 
- under `src/cfgs_adroit/task/relocate.yaml` you will see that relocate has `encoder_lr_scale: 0.01`, as shown in the paper, relocate requires a smaller encoder learning rate. You can set specific default parameters for each task in their separate config files. 
- in the paper for most experiments, I used `frame_stack=3`, however later I found we can reduce it to 1 and still get the same performance. It might be beneficial to set it to 1 so it runs faster and takes less memory. If you set this to 1, then convolutional channel expansion will only be applied for the relocate env, where the input is a stack of 3 camera images. 
- all values in table 2 in appendix A.2 of the paper are set to be the default values in the config files. 

## Computation time
This table compares the computation time estimates for the open source code with default hyperparameters (tested on NYU Greene with RTX 8000 and 4 cpus). When you use the code on your machine, it might be slightly faster or slower, but should not be too different. These results seem to be slightly faster than what we reported in the paper (which tested on Azure P100 GPU machines). Improved computation speed is mainly due to we now set default `frame_stack` for Adroit.

| Task  | Stage 2 (30K updates) | Stage 3 (4M frames) | Total   | Total (paper) | 
|------------------|-----------------------|---------------------|---------|------------|
| Door/Pen/Hammer  | ~0.5 hrs              | ~13 hrs             | ~14 hrs | ~16 hrs         |
| Relocate         | ~0.5 hrs              | ~16 hrs             | ~17 hrs |   ~24 hrs       |

Note that VRL3's performance kind of converged already at 1M data for Door, Hammer and Relocate. So depending on what you want to achieve in your work, you may or may not need to run a full 4M frames. In the paper we run to 4M to be consistent with prior work and show VRL3 can outperform previous SOTA in both short-term and long-term performance. 

## Known issues:
- Some might encounter a problem where mujoco can crush at an arbitrary point during training. I have not seen this issue before but I was told reinit `self.train_env` between stage 2 and stage 3 can fix it. 
- If you are not using the provided docker image and you run into the problem of slow rendering, it is possible that mujoco did not find your gpu and made a `CPUExtender` instead of a `GPUExtender`. You can follow the steps in the provided dockerfile, or force it to use the `GPUExtender` (see code in `mujoco-py/mujoco_py/builder.py`) Thanks to ZheCheng Yuan for identifying above 2 issues. 
- Newer versions of mujoco are easier to work with. We use an older version only because Adroit relies on it. (So you can try a newer mujoco if you want to test on other environments). 

## Acknowledgement
VRL3 has been mainly built on top of the DrQv2 codebase (https://github.com/facebookresearch/drqv2). 

## Citation
If you use VRL3 in your research, please consider citing the paper as:
```
@inproceedings{wang2022vrl3,
  title={VRL3: A Data-Driven Framework for Visual Deep Reinforcement Learning},
  author={Wang, Che and Luo, Xufang and Ross, Keith and Li, Dongsheng},
  booktitle={Conference on Neural Information Processing Systems},
  year={2022},
  url={https://openreview.net/forum?id=NjKAm5wMbo2}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
