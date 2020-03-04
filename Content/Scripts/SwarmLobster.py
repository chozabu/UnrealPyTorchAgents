import unreal_engine as ue

import math
import random
import torch
import gym
import numpy as np
from TD3 import TD3
from utils import ReplayBuffer
import time
import os
import copy
from pathlib import Path
import threading

from torch.utils.tensorboard import SummaryWriter

from unreal_engine import FVector, FRotator, FTransform, FHitResult, FLinearColor
from unreal_engine.classes import ActorComponent, ForceFeedbackEffect, KismetSystemLibrary, WidgetBlueprintLibrary
from unreal_engine.enums import EInputEvent, ETraceTypeQuery, EDrawDebugTrace

max_action = 1.0

# gamma = 0.99  # discount for future rewards
# batch_size = 128  # num of transitions sampled from replay buffer
# lr = 0.00001
# exploration_noise = 0.4
# polyak = 0.995  # target policy update parameter (1-tau)
# policy_noise = 0.2  # target policy smoothing noise
# noise_clip = 0.5
# policy_delay = 2  # delayed policy updates parameter
# max_timesteps = 1000  # max timesteps in one episode

directory = "./NNModels/FaceLobster3d"  # save trained models

Path(directory).mkdir(parents=True, exist_ok=True)

print(os.path.abspath(directory))

master = None

class TorchWalkerMaster:
    def __init__(self):
        global master
        self.has_init = False
        master = self



        self.replay_buffer = ReplayBuffer(max_size=200000)

        self.frame = 0

        self.episode = 0
        self.worker_id = 0

        self.can_thread = True
        self.actor = None

    # this is called on game start
    def begin_play(self):
        global master
        if not master:
            master = self

        if not self.actor:
            self.actor = self.uobject.get_owner()

        ue.log('Begin Play on TorchWalkerMaster class')
        ue.log("Has CUDA: {}".format(torch.cuda.is_available()))


        self.writer = SummaryWriter(os.path.join(directory, self.actor.SaveName))

    def init_network(self, state_dim, action_dim):
        if not self.has_init:
            if not self.actor:
                self.actor = self.uobject.get_owner()
            print("--INITNET---")
            self.has_init = True
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.policy = TD3(self.actor.learning_rate, state_dim, action_dim, max_action)
            print(state_dim)
            print(action_dim)
            print("--INITNET---")

            if self.actor.LoadFile:
                self.policy.load(directory, self.actor.LoadName)

    def get_next_ep(self):
        self.episode += 1
        return self.episode
    def get_id(self):
        retid = self.worker_id
        self.worker_id += 1
        return retid
    def write_data(self,ep_reward, ep_reward_avg, ep_frame):
        real_ep = self.episode
        self.writer.add_scalar('ep_reward',
                               ep_reward,
                               real_ep)
        self.writer.add_scalar('ep_avg_reward',
                               ep_reward_avg,
                               real_ep)
        self.writer.add_scalar('ep_frame',
                               ep_frame,
                               real_ep)
        #print("finished ep {}, avgscore: {}".format(real_ep, ep_reward_avg))
        self.episode += 1
    def transfer_buffer(self, buffer):
        self.replay_buffer.mergein(buffer)
        #print("buffer merged, length: {}".format(self.replay_buffer.size))

    def thread_func(self):
        if self.replay_buffer.size:
            al, c1l, c2l, prl = self.policy.update(self.replay_buffer, 200, self.actor.batch_size, self.actor.gamma, self.actor.polyak, self.actor.policy_noise,
                                                   self.actor.noise_clip, self.actor.policy_delay)
            print("aloss:{}, frame:{}, mem:{}".format(al, self.frame, self.replay_buffer.size))
            self.writer.add_scalar('actor_loss',
                                   al,
                                   self.frame)
            self.writer.add_scalar('c1_loss',
                                   c1l,
                                   self.frame)
            self.writer.add_scalar('c2_loss',
                                   c2l,
                                   self.frame)

        else:
            print("skipping")
            time.sleep(0.01)
        self.can_thread = True

    def tick(self, delta_time):
        self.frame += 1

        if self.replay_buffer.size < 10000:
            return

        if self.can_thread:
            x = threading.Thread(target=self.thread_func)#, args=(1,))
            x.start()
            self.can_thread = False

        if self.frame % 600 == 0:
            self.policy.save(directory, self.actor.SaveName)

class TorchWalkerMinion:

    def __init__(self):
        self.replay_buffer = ReplayBuffer(max_size=50000)
        self.last_state = []
        self.last_action = None
        self.last_done = False

        self.episode = 0

        self.ep_frame = 0
        self.ep_reward = 0

        self.random_frames = 10

        self.exploration_noise = random.random()*0.3

    # this is called on game start
    def post_init(self):
        self.actor = self.uobject.get_owner()
        ue.log('Begin Play on TorchWalkerMinion class')

        print("MASTER")
        print(master)

        actionlen = self.actor.get_action_dim()
        obslen = len(self.actor.update_observation()[0])
        master.init_network(obslen, actionlen)

        self.my_id = master.get_id()

        self.policy = master.policy

        self.action_space_low = [-1 for x in range(master.action_dim)]
        self.action_space_high = [1 for x in range(master.action_dim)]

        self.obs_space_low = [-1 for x in range(master.state_dim)]
        self.obs_space_high = [1 for x in range(master.state_dim)]

    def reset_ep(self):
        self.ep_frame = 0
        self.ep_reward = 0
        self.episode += 1
        self.actor.reset_dude()
        self.random_frames = random.randint(3,9)

    def pytick(self):
        self.ep_frame+=1

        #############get observation#############
        obs = self.actor.update_observation()[0]
        state = np.array(obs)
        state = state.clip(self.obs_space_low, self.obs_space_high)

        ### random action for a few frames ###
        if self.ep_frame < self.random_frames:
            action = np.random.normal(0, master.actor.exploration_noise*5, size=master.action_dim)
            self.actor.set_action(action.tolist())
            return

        #############get action from policy###############
        action = self.policy.select_action(state)
        action = action + np.random.normal(0, self.exploration_noise, size=master.action_dim)
        action = action.clip(self.action_space_low, self.action_space_high)

        self.actor.set_action(action.tolist())

        ########## calc reward ###########
        reward = self.actor.calc_reward()

        #### eval critic for debugging info ###
        if random.random()>.9: #  not every frame for performance
            self.actor.action_eval = self.policy.eval_action(state, action)

        done = self.actor.done
        #timeout, new EP
        if self.ep_frame > max_timesteps:
            done = True

        self.ep_reward += reward

        ####### record action ############
        if self.ep_frame > self.random_frames-1:
            self.replay_buffer.add((self.last_state, self.last_action, reward, state, float(done)))
        self.last_state = state
        self.last_action = action

        if done:
            self.actor.done = False
            ep_reward_avg = self.ep_reward/self.ep_frame
            master.write_data(self.ep_reward, ep_reward_avg, self.ep_frame)
            self.reset_ep()
            master.transfer_buffer(self.replay_buffer)
            self.replay_buffer = ReplayBuffer()


