# UnrealPytorchAgents

This is a project trying out RL with pytorch and ue4

## setup

uses master branch of UnrealEnginePython https://github.com/20tab/UnrealEnginePython
clone it into the projects Plugins folder (this should really be a sub-repo)

in windows, using global python env, packages:  
tensorflow-tensorboard==0.4.0rc3  
pytorch, using command from pytorch website  
more?

Commands to get setup may be something like

```
git clone https://github.com/chozabu/UnrealPyTorchAgents
cd https://github.com/chozabu/UnrealPyTorchAgents
mkdir Plugins
cd Plugins
git clone git@github.com:20tab/UnrealEnginePython.git

```

## usage

Files are currently saved to a folder like "/Epic Games/UE_4.23/Engine/Binaries/Win64/NNModels"

## video

https://i.gyazo.com/803b7bd236e042cf321c58652eaa4428.mp4 - bipedal walker, doing fairly well

## info

training is done "on the fly" in a background thread from physical instances of an agent, all sharing the same neural network

Network is TD3 based on this implementation: https://github.com/nikhilbarhate99/TD3-PyTorch-BipedalWalker-v2

## interesting reads/projects 

https://spinningup.openai.com/en/latest/  
Reinforcement Learning: An Introduction. Second edition, in progress. Richard S. Sutton and Andrew G. Barto  
unity-ml-agents




