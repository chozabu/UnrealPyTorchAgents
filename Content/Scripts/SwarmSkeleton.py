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


# joint_obs_num = 18
# state_dim = joint_obs_num
# state_dim += 2 #pitch/roll
# state_dim += 1 #ground dist
# state_dim += 7*6 #ang/lin vel
# state_dim += 7*6 #pos/rot
# state_dim += 8 # ground sensors
# state_dim += 2 # contact sensors
# state_dim += 1 # target angle - calculated in PY
#
#
# action_dim = 14

#state_dim += action_dim

max_action = 1.0

max_power = 60000
trace_length = 100 * 40

log_interval = 60  # print avg reward after interval
gamma = 0.99  # discount for future rewards
batch_size = 128  # num of transitions sampled from replay buffer
lr = 0.00001
exploration_noise = 0.4
polyak = 0.995  # target policy update parameter (1-tau)
policy_noise = 0.2  # target policy smoothing noise
noise_clip = 0.5
policy_delay = 2  # delayed policy updates parameter
max_timesteps = 1000  # max timesteps in one episode

directory = "./NNModels/SkelWalker3d"  # save trained models
#filename = "TD3_BLIND"
loadpol = False
loadfilename = "TD3_Skel3dX4"
filename =     "TD3_Skel3dX4"
#TD3_BipedalWalker3dComplex2 angdamp = 600
#TD3_BipedalWalker3dComplex2a angdamp = 100
#TD3_BipedalWalker3dComplex2b angdamp = 20


Path(directory).mkdir(parents=True, exist_ok=True)

print(os.path.abspath(directory))


master = None

class TorchWalkerMaster:

    # this is called on game start
    def begin_play(self):
        global master
        self.has_init = False
        master = self
        self.replay_buffer = ReplayBuffer(max_size=200000)
        ue.log('Begin Play on TorchWalkerMaster class')
        ue.log("Has CUDA: {}".format(torch.cuda.is_available()))


        self.frame = 0


        self.episode = 0
        self.worker_id = 0

        self.writer = SummaryWriter(os.path.join(directory, filename))
        self.can_thread = True

    def init_network(self, state_dim, action_dim):
        if not self.has_init:
            print("--INITNET---")
            self.has_init = True
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.policy = TD3(lr, state_dim, action_dim, max_action)
            print(state_dim)
            print(action_dim)
            print("--INITNET---")

            if loadpol:
                self.policy.load(directory, loadfilename)

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
            al, c1l, c2l, prl = self.policy.update(self.replay_buffer, 200, batch_size, gamma, polyak, policy_noise,
                                                   noise_clip, policy_delay)
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

    def thread_func_crit(self):
        if self.replay_buffer.size:
            al, c1l, c2l, prl = self.policy.update(self.replay_buffer, 200, batch_size, gamma, polyak, policy_noise,
                                                   noise_clip, policy_delay)
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
            # if self.can_thread:
            #     x = threading.Thread(target=self.thread_func_crit)#, args=(1,))
            #     x.start()
            #     self.can_thread = False
            # print("buffer size: {}".format(self.replay_buffer.size))
            return

        if self.can_thread:
            x = threading.Thread(target=self.thread_func)#, args=(1,))
            x.start()
            self.can_thread = False

        if self.frame % 600 == 0:
            self.policy.save(directory, filename)

class TorchWalkerMinion:

    # this is called on game start
    def begin_play(self):
        self.actor = self.uobject.get_owner()
        self.replay_buffer = ReplayBuffer(max_size=50000)
        ue.log('Begin Play on TorchWalkerMinion class')

        #self.policy = TD3(lr, state_dim, action_dim, max_action)
        self.gen_target()

        self.last_state = []
        self.last_reward = 0
        self.last_action = None
        self.last_done = False
        self.frame = int(random.random() * 100)
        self.start_pos = self.uobject.get_actor_location()


        self.episode = 0

        self.ep_frame = 0
        self.ep_reward = 0
        self.total_frame = 0


        self.boredom = 0.8

        print("MASTER")
        print(master)

        actionlen = self.actor.get_action_dim()
        TEMP_OBS = self.actor.update_observation()[0]
        print("TEMP_OBS")
        print(TEMP_OBS)
        obslen = len(TEMP_OBS)
        print(obslen)
        master.init_network(obslen+1, actionlen)

        self.my_id = master.get_id()
        #self.actor.TextRender.call('SetText {}'.format(self.my_id))

        self.random_frames = 10

        self.bg_thread = None

        self.exploration_noise = random.random()*0.3
        self.first_frame = True



        self.policy = master.policy

        self.action_space_low = [-1 for x in range(master.action_dim)]
        self.action_space_high = [1 for x in range(master.action_dim)]


        self.obs_space_low = [-1 for x in range(master.state_dim)]
        self.obs_space_high = [1 for x in range(master.state_dim)]

    def gen_target(self):
        target_angle = math.pi*.5# random.random() * math.pi * 2.0
        target_dist = 1000+random.random() * 100 * 1500
        self.target_x = math.cos(target_angle) * target_dist
        self.target_y = math.sin(target_angle) * target_dist
        self.last_dist = self.get_target_dist()

    def get_target_angle(self):
        location = self.actor.get_actor_location()
        angle = self.actor.GetTargetAngleFromPos(FVector(self.target_x, self.target_y, location.z))[0]
        nangle = angle / 360 + .5
        return nangle

    def get_angle_rads(self):
        angle = self.actor.get_actor_rotation().yaw
        rangle = angle / 180 * math.pi
        return rangle

    def get_target_dist(self):
        location = self.actor.get_actor_location()
        xd = location.x - self.target_x
        yd = location.y - self.target_y
        #perhaps check furthest component instead?
        #foreach component:
            #get biggest dstsqr
        #return sqrt(biggest)
        return math.sqrt(xd * xd + yd * yd)

    def reset_ep(self):
        #print("reset")
        self.boredom = 0.8
        self.ep_frame = 0
        self.ep_reward = 0
        self.episode += 1
        self.actor.reset_dude()
        #self.actor.ResetPos()
        self.gen_target()
        self.random_frames = random.randint(3,9)
        self.first_frame = True

    def tick(self, delta_time):
        self.ep_frame+=1
        self.total_frame += 1

        #############get observation#############
        obs = self.actor.update_observation()[0]

        target_angle = self.get_target_angle()
        obs.append(target_angle)

        state = np.array(obs)
        state = state.clip(self.obs_space_low, self.obs_space_high)

        #############get action from policy###############
        if self.total_frame > 0:
            if self.ep_frame < self.random_frames:
                action = np.random.normal(0, exploration_noise*5, size=master.action_dim)
            else:
                action = self.policy.select_action(state)
            action = action + np.random.normal(0, self.exploration_noise, size=master.action_dim)
            action = action.clip(self.action_space_low, self.action_space_high)
        else:
            action = np.random.normal(0, self.exploration_noise*.2, size=master.action_dim)


        ###############apply action################
        self.actor.set_action(action.tolist())

        ########## calc reward ###########
        reward = self.actor.calc_reward()

        #reward for keeping joints closer to neutral
        rv = 0
        for jo in self.actor.joint_obs:
            rv += abs(jo-0.5)-.1
        #print(-rv)
        rv *= .01
        reward -= rv

        reward -= abs(target_angle)-0.7


        #getting closer to target
        new_dist = self.get_target_dist()
        diffd = self.last_dist - new_dist
        reward += diffd*.5

        self.uobject.draw_debug_line(self.actor.get_actor_location(),
                                     FVector(self.target_x, self.target_y, self.actor.get_actor_location().z),
                                     FLinearColor(0, 1, 0),
                                     0, 3)

        self.last_dist = new_dist

        #staying away from ground
        reward += self.actor.ground_dist-.2
        reward += self.actor.head_ground_dist-.2

        if random.random()>.9:
            self.actor.action_eval = self.policy.eval_action(state, action)

        #timeout, new EP
        done = 0
        if self.ep_frame > max_timesteps:
            done = 1

        if self.actor.ground_dist < .045:
            self.actor.hit_body = False
            done = 1
            reward -= 30
        if self.actor.head_ground_dist < .085:
            self.actor.hit_body = False
            done = 1
            reward -= 30


        # if 1:#self.actor.get_display_name() == "BipedalWalker3d":
        #     print(" --- ")
        #     print(" --- ")
        #     print(self.replay_buffer.size)
        #     print(" --- ")
        #     print("joints")
        #     print(-rv)
        #     print("feet touching")
        #     print(2-self.actor.rfoot_touch-self.actor.lfoot_touch)
        #     print("ground dist")
        #     print(self.actor.ground_dist-.3)
        #     # print("balance")
        #     # print((self.actor.body.get_up_vector().z-.4)*.5)
        #     print("total")
        #     print(reward)

        self.ep_reward += reward

        self.actor.last_reward = reward

        ####### record action ############
        if self.ep_frame > self.random_frames-1:
            self.replay_buffer.add((self.last_state, self.last_action, reward, state, float(done)))
        self.last_state = state
        self.last_action = action

        if done:
            ep_reward_avg = self.ep_reward/self.ep_frame
            master.write_data(self.ep_reward, ep_reward_avg, self.ep_frame)
            self.reset_ep()
            master.transfer_buffer(self.replay_buffer)
            self.replay_buffer = ReplayBuffer()

        # if new_dist < 500:
        #     self.gen_target()
        #     self.last_dist = self.get_target_dist()
        #     #ue.log("NEW TARGET!!!!!")

        self.frame += 1
        self.actor.rfoot_touch = 0
        self.actor.lfoot_touch = 0
        #if self.my_id == 1:
        #print(state)

