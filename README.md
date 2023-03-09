# VRL3 codebase
Official code by VRL3 authors. This codebase is currently undergoing clean up and will be updated. 

**Please do not** post this code online for now!!! The public official repo is still being processed. 

I rewrote the entire codebase to make it cleaner and improve readability, so it's possible sth might break, if you run into issues or cannot reproduce the paper results, please don't hesitate to send me an email! (Also, the paper has too many ablations, the code for most of them are now deleted from the codebase so it's not too long and complicated, but if you want to know any details feel free to ask me.)

Repo structure and important files: 
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

vrl3data # download this folder from the google drive link
└───demonstrations # adroit demos 
└───trained_models # pretrained stage 1 models
```

## Download adroit demos and pretrained models


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
Once you get into the container (either docker or singularity), first run the following commands so the paths are correct. Very important especially on singularity since it uses automount which can mess up the paths.  
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

## Some hyperparameter details
- BC loss: in the config files, I now by default disable all BC loss since our ablations show they are not really helping. 
- under `src/cfgs_adroit/task/relocate.yaml` you will see that relocate has `encoder_lr_scale: 0.01`, as shown in the paper, relocate requires a smaller encoder learning rate. You can set specific default parameters for each task in their separate config files. 
- in the paper for most experiments, I used `frame_stack=3`, however later I found we can reduce it to 1 and still works with the same performance. It might be beneficial to set it to 1 so it runs faster and takes less memory. If you set this to 1, then convolutional channel expansion will only be applied for for the relocate env, where the input is a stack of 3 camera images. 
- all values in table 2 in appendix A.2 of the paper are set to be the default values in the config files. 

## Computation time
This table compares the computation time estimates for the open source code with default hyperparameters (tested on NYU Greene with RTX 8000 and 4 cpus). When you use the code on your machine, it might be slightly faster or slower, but should not be too different. These results seem to be slightly faster than what we reported in the paper (which used an older version of the code, and tested on Azure P100 GPU machines).

| Task  | Stage 2 (30K updates) | Stage 3 (4M frames) | Total   | Total (paper) | 
|------------------|-----------------------|---------------------|---------|------------|
| Door/Pen/Hammer  | ~0.5 hrs              | ~13 hrs             | ~14 hrs | ~16 hrs         |
| Relocate         | ~0.5 hrs              | ~16 hrs             | ~17 hrs |   ~24 hrs       |

Note that VRL3's performance kind of converged already at 1M data for Door, Hammer and Relocate. So depending on what you want to achieve in your work, you may or may not need to run a full 4M frames. In the paper we run to 4M to be consistent with prior work and show VRL3 can outperform previous SOTA in both short-term and long-term performance. 

## reproduce plots
Essentially all experiment logs are in the logs folder form the drive link. And then I basically run `plot_utils/vrl3_plot_runner.py` to generate all the figures. (it uses plotting helper functions in `vrl3_plot_helper.py`) But there were too many figures, and I still need some time to clean up the plotting code so it might be difficult to use for now. Sorry for that.

Still working on it now... Should allow other people to easily reproduce all paper figures with 1-click.

### below: STILL WORKING ON IT NOW...
# Project

> This repo has been populated by an initial template to help get you started. Please
> make sure to update the content to build a great experience for community-building.

As the maintainer of this project, please make a few updates:

- Improving this README.MD file to provide a great experience
- Updating SUPPORT.MD with content about this project's support experience
- Understanding the security reporting process in SECURITY.MD
- Remove this section from the README

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
