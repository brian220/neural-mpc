# Neural MPC Ablation Study
This team implementaion project in the Reinforcement Learning course in NCTU.<br />
We do the ablation study on the paper "Model-based Reinforcement Learning with Neural Network Dynamics" -  https://arxiv.org/pdf/1708.02596.pdf <br />
And This repo is fork from https://github.com/aravindsrinivas/neural-mpc, which finished the model-base part of the paper. <br />
We train the model and do some ablation study on the model-based model.<br />
And we planed to implement the model-free part of the paper, but we meet some problem in the implementation of PPO part,
so if you find out which part go wrong in our code, please contact us, we will very appreciate about that.

# Dependency
We also provide a ```requirement.txt```, hope this can help you if you want to run this code.<br />
To run the code in the CPU mode, ```pip intall -r requirement.txt```, and please use the python version <= 3.6. (we use 3.6)<br />
To run the code in the GPU mode, you need to install CUDA 8.0 and cudnn 6.0 <br />

# Results
Some results we get: <br />

![image](https://github.com/brian220/neural-mpc/blob/master/images/objective.png)


![image](https://github.com/brian220/neural-mpc/blob/master/images/Aggregation.png)

![image](https://github.com/brian220/neural-mpc/blob/master/images/K.png)

![image](https://github.com/brian220/neural-mpc/blob/master/images/H.png)



