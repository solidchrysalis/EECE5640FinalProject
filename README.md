# Final Project for EECE5640

## Starting Steps

Create a conda environment and download the necesary packages:

```bash
python -m venv venv
pip install pytorch torchvision 
```

Alternatively, activate the environment in my directory:

```bash
cd /home/rowan.t/ThomasRowan_FinalProject
source /home/rowan.t/miniconda/bin/activate
conda activate myenv
```

## Make code

To make and install the code, run

```bash
make all ARCH=[architecture]
bash install.bash
```

Architectures are: sm_60, sm_70

## Run code

To run the code perform:

```bash
python3 cnn.py [dataset] [optimizer]
```

Datasets are: cifar, fashionmnist
Optimizers are: adam, adagrad, stochastic

Alternatively, submit P100 and V100 as an sbatch script to run all possible commbinations.