## Generating balanced multiplayer game levels with GANs and RL
The main idea of this project was to automatically, with as little input as possible, generate game levels. As a level generation is a rather broad area we focused on a more clearly defined goal. And we decided to train a generator capable of generating boards that are _fair_ for all the players involved in the game.

As a proof of concept, a simple racing game was chosen to be used to evaluate if algorithms work. More details are available in [my thesis](https://drive.google.com/file/d/1ypxK8KeoR3lBjRLUbAzpL4O2isyZhATc/view?usp=sharing).

## Concept
![diagram of a system from the poster](/images/concept-diagram.svg)


## Evaluation & Results
![discriminator loss](/images/discriminator-loss.svg)
![generator loss and tracks](/images/generator-training.svg)

---

### How to run
To run experiments for yourself, you can execute the following commands.

#### Train agents
```bash
python train-agents.py
```
This command will run training on a few random predefined tracks. When agents are sufficiently trained, which will probably take quite a lot of time, you can stop the script.

#### Train generator and discriminator
```bash
python train-gan.py --agents=<path_to_dir_with_trained_agents>
```

#### Play with generator
With the trained generator you can evaluate whether the tracks it generates are more fair:
```bash
python evaluate.py --generator=<path_to_dir_with_generator> \
                   --agents=<path_to_dir_with_trained_agents>
```

You can also explore how the latent space looks like using the following command (this script requires the `pyforms==3.0.0` to be installed):
```bash
python explore.py --generator=<path_to_dir_with_generator>
```

Or run latent interpolation:
```bash
python latent-interpolation.py --generator=<path_to_dir_with_generator>
```
