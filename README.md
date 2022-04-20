# Fork Changes
- Refactored code
- Added dependencies in environment.txt (**but don't use it for creating new env** it contains dependencies carried over from a last project)
- Added dropout layers + config and weight decay.
- Added a script to generate commands multiple configs
- Removed wandb loggers and added custom logger
- Added class imbalanced training
- Added Curriculum learning with pretrained ResNet18


## Example config

```python train.py --batchsize 1028 --workers 2 --epochs 80 --opt adam --arch 'mlp[16384,16384,512]' --reg wd --iid```

See `exps.sh` more details.

# deep-bootstrap-code

Code for the paper [The Deep Bootstrap Framework: Good Online Learners are Good Offline Generalizers](https://arxiv.org/abs/2010.08127).

The main training code is [here](/inftrain/train.py), and a sample configuration of hyperparameter sweep (using [Caliban](https://github.com/google/caliban)) is [here](/inftrain/sample_sweep.json).

The CIFAR-5m dataset is released at: https://github.com/preetum/cifar5m
