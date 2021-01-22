# Pulsar
Pulsar is a research framework for training intelligent agents via reinforcement learning to compete in Robomaster's (ICRA) AI challenge. The framework is divided into three major parts: rmleague for the training scheme, Truly + Distributed PPO as the reinforcement learning algorithm, and a physics simulation to act as the simulation of the game.

# Demo of the simulation
![](https://github.com/HKU-ICRA/Pulsar/tree/master/videos/pulsar_demo1.gif)
![](https://github.com/HKU-ICRA/Pulsar/tree/master/videos/pulsar_demo2.gif)

# Video (technical proposal)
[![](http://img.youtube.com/vi/66CMskieKAU/0.jpg)](http://www.youtube.com/watch?v=66CMskieKAU "")

# Network architecture
![Pulsar](https://github.com/HKU-ICRA/Pulsar/blob/master/architecture/pulsar_architecture.png)

# Network framework per agent
![Pulsar_framework_agent](https://github.com/HKU-ICRA/Pulsar/blob/master/architecture/pulsar_framework_agent.png)

# rmleague framework
![Pulsar_framework_process](https://github.com/HKU-ICRA/Pulsar/blob/master/rmleague/pulsar_framework_process.png)

# Citations
* The network's architecture as well as training method is heavily inspired by Deepmind's Alpha-star:
>Vinyals, O., Babuschkin, I., Czarnecki, W.M. et al. Grandmaster level in StarCraft II using multi-agent reinforcement learning. Nature
>(2019) doi:10.1038/s41586-019-1724-z

* Training algorithm:
>@misc{wang2019truly,
>    title={Truly Proximal Policy Optimization},
>    author={Yuhui Wang and Hao He and Chao Wen and Xiaoyang Tan},
>    year={2019},
>    eprint={1903.07940},
>    archivePrefix={arXiv},
>    primaryClass={cs.LG}
>}


