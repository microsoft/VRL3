# VRL3 codebase
Official code by VRL3 authors. This codebase is currently undergoing clean up and will be updated. 

Repo structure and important files: 
```
VRL3
│   README.md # read this file first!
└───docker # dockerfile with all dependencies
└───plot_utils # code for plotting, still working on it now...
└───src
    │   train_adroit.py # setup and main training loop
    │   vrl3_agent.py # agent class, code for stage 2, 3
    │   train_stage1.py # code for stage 1 pretraining on imagenet
    │   stage1_models.py # the encoder classes pretrained in stage 1
    └───cfgs_adroit # configuration files with all hyperparameters

vrl3data # download this folder from the google drive link (450 mb)
└───demonstrations # adroit demos 
└───trained_models # pretrained stage 1 models
```

## Set up environment
The recommended way is to just use the dockerfile I provided and follow the tutorial here. You can also look at the dockerfile to know the exact dependencies or modify it to build a new dockerfile. 

### Run with docker
If you have a local machine with gpu, or your cluster allows docker (you have sudo), then you can just pull my docker image and run code there. 

```
docker pull docker://cwatcherw/vrl3:1.2
```

Now, `cd` into a directory where you have the `VRL3` folder (this repo), and also the `vrl3data` folder that you downloaded from my google drive link. 
Then, mount `VRL3/src` to `/code`, and mount `vrl3data` to `/vrl3data` (you can also mount to other places, but you will need to adjust some commands or paths in the config files):
```
docker run -it --rm --gpus all -v "$(pwd)"/VRL3/src:/code -v "$(pwd)"/vrl3data:/vrl3data  docker://cwatcherw/vrl3:1.2
```

After the container starts, run following commands: 
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=/workspace/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/.mujoco/mujoco210/bin
export MUJOCO_GL=egl
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


### Run with singularity 
If your cluster does not allow sudo (for example, NYU's slurm HPC), then you can use singularity, it is similar to docker.

Set up singularity container:
```
cd /scratch/$USER/sing/
singularity build --sandbox vrl3 docker://cwatcherw/vrl3:1.2
```

For example, on NYU HPC, start interactive session (if your school has a different hpc system, consult your hpc admin): 
```
srun --pty --gres=gpu:1 --cpus-per-task=4 --mem 12000 -t 0-06:00 bash

cd /scratch/$USER/sing
singularity exec --nv -B /scratch/$USER/sing/VRL3/src:/code -B /scratch/$USER/sing/vrl3sing/opt/conda/lib/python3.8/site-packages/mujoco_py/:/opt/conda/lib/python3.8/site-packages/mujoco_py/ -B /scratch/$USER/sing/vrl3data:/vrl3data /scratch/$USER/sing/vrl3sing bash
```
Here, by default VRL3 uses 4 workers for dataloader, so we request 4 cpus. We also mount the `mujoco_py` package folder because singularity files by default are read-only, and the older version of mujoco_py wants to modify files, which can be problematic. (And Adroit env relies on older version of mujoco so we have to deal with it...)

After getting into the singularity container, run the following so python knows where to find mujoco files (this is needed because singulairty does some automount that can mess up the paths):
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=/workspace/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/.mujoco/mujoco210/bin
export MUJOCO_GL=egl
```

To train VRL3 on Adroit:
```
cd /code
python train_adroit.py task=door
```
or use debug option to do faster test runs. 

## reproduce plots
Should allow other people to reproduce all paper figures with 1-click. 
Still working on it now...

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
