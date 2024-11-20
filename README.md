
# RL_Games
This repository contains Proximal Policy Optimization (PPO)-based reinforcement learning (RL) agents trained to play Mortal Kombat 3 and Street Fighter 2. The agents are capable of completing Round 1 of each game after significant training, showcasing their ability to learn strategies.

## Environment Setup

All packages and their versions have been specified in ``` package_list.txt```


Git clone the repository using the following command:

```git clone https://github.com/SumukhShrote/RL_Games.git```

## Street Fighter 2

![StreetFighterVid](StreetFighter/assets/StreetFighter.mp4)

1. Go to the roms directory ```./StreetFighter/roms``` and place the game rom in the directory (of a file type supported by gym-retro package)


2. Open the terminal in the same directory and run the following command:
```python3 -m retro.import .```


3. To train and test the agent, run the jupyter notebook 
```./StreetFighter/PPOTutorial.ipynb```

4. The agent begins to beat the level 1 of the game after around 280000 steps of training


## Mortal Kombat 3

![MortalKombatVid](MortalKombat/assets/MortalKombat.mp4)

1. Go to the roms directory ```./Mortal Kombat/roms``` and place the game rom in the directory (of a file type supported by gym-retro package)


2. Open the terminal in the same directory and run the following command:
```python3 -m retro.import .```


3. To train the agent, run the following command: 
```python3 ./Mortal Kombat/train.py```

4. To test the agent, run the following command: 
```python3 ./Mortal Kombat/test.py```

5. The agent begins to beat the level 1 of the game after around 200000 steps of training

