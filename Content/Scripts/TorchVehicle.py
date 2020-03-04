import unreal_engine as ue

ue.log('Hello i am a Python module')

import math
import random
import torch
import gym
import numpy as np
from TD3 import TD3
from utils import ReplayBuffer
import time
import os

from torch.utils.tensorboard import SummaryWriter

from unreal_engine import FVector, FRotator, FTransform, FHitResult, FLinearColor
from unreal_engine.classes import ActorComponent, ForceFeedbackEffect, KismetSystemLibrary, WidgetBlueprintLibrary
from unreal_engine.enums import EInputEvent, ETraceTypeQuery, EDrawDebugTrace

trace_num = 8
state_dim = trace_num + 1 + 1 + 1 + 1  # trace+speed+angle+dist+steer
action_dim = 3
max_action = 1.0

max_power = 60000
trace_length = 100 * 20

log_interval = 60  # print avg reward after interval
gamma = 0.99  # discount for future rewards
batch_size = 100  # num of transitions sampled from replay buffer
lr = 0.001
exploration_noise = 0.3
polyak = 0.995  # target policy update parameter (1-tau)
policy_noise = 0.2  # target policy smoothing noise
noise_clip = 0.5
policy_delay = 2  # delayed policy updates parameter
max_timesteps = 2000  # max timesteps in one episode

directory = "./NNModels/AutoCar"  # save trained models
#filename = "TD3_BLIND"
loadfilename = "TD3_LIDAR8hardcore2"
filename = "TD3_LIDAR8hardcore3"

from pathlib import Path

Path(directory).mkdir(parents=True, exist_ok=True)

print(os.path.abspath(directory))

# env = gym.make("BipedalWalker-v2")
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
# max_action = float(env.action_space.high[0])

class TorchDriver:

    # this is called on game start
    def begin_play(self):
        self.actor = self.uobject.get_owner()
        self.VehicleMovement = self.actor.VehicleMovement
        self.replay_buffer = ReplayBuffer(max_size=50000)
        ue.log('Begin Play on TorchActor class')
        ue.log(torch.cuda.is_available())
        ue.log(dir(self.uobject))

        self.policy = TD3(lr, state_dim, action_dim, max_action)
        self.gen_target()

        self.last_state = []
        self.last_reward = 0
        self.last_action = None
        self.last_done = False
        self.frame = int(random.random() * 100)
        self.start_pos = self.uobject.get_actor_location()

        # self.actor.AutoDrive = True
        self.policy.load(directory, loadfilename)
        self.filename = "{}_{}".format(filename, self.frame)

        self.episode = 0

        self.writer = SummaryWriter(os.path.join(directory, filename))

        self.ep_frame = 0
        self.ep_reward = 0

        self.ep_reward_avg_BEST = 0
        # self.gstate = env.reset()##BIPEDAL
        # self.gdone = False
        # self.greward = 0

    def gen_target(self):
        target_angle = random.random() * math.pi * 2.0
        target_dist = random.random() * 100 * 2000
        self.target_x = math.cos(target_angle) * target_dist
        self.target_y = math.sin(target_angle) * target_dist
        self.last_dist = self.get_target_dist()

    def get_target_angle(self):
        location = self.actor.get_actor_location()
        angle = self.actor.GetTargetAngleFromPos(FVector(self.target_x, self.target_y, location.z))[0]
        nangle = angle / 360 + .5
        # xd = location.x-self.target_x
        # yd = location.y-self.target_y
        # rangle = math.atan2(yd,xd)
        # nangle = rangle+math.pi
        # nangle = nangle/(math.pi*2)
        return nangle

    def get_angle_rads(self):
        angle = self.actor.get_actor_rotation().yaw
        rangle = angle / 180 * math.pi
        return rangle

    def get_target_dist(self):
        location = self.actor.get_actor_location()
        xd = location.x - self.target_x
        yd = location.y - self.target_y
        return math.sqrt(xd * xd + yd * yd)

    def reset_ep(self):
        self.ep_frame = 0
        self.ep_reward = 0
        self.episode += 1
        self.actor.ResetPos()
        self.gen_target()

    # this is called at every 'tick'
    def tick(self, delta_time):
        #############get observation#############
        # ue.log(vel)
        self.ep_frame+=1
        obs = [self.VehicleMovement.GetForwardSpeed() / 6000 + 0.5]  # vel.x/1000+.5, vel.y/1000+.5]

        # LIDAR
        location = self.uobject.get_actor_location() + FVector(0, 0, 40)
        if (trace_num):
            rangle = self.get_angle_rads()
            for trace_id in range(trace_num):
                angle = trace_id / trace_num * math.pi * 2.0 + rangle
                ltarget = location + FVector(math.cos(angle) * trace_length, math.sin(angle) * trace_length, 0)
                is_hitting_something, _, hit_result = KismetSystemLibrary.LineTraceSingle(self.uobject,
                                                                                          location,
                                                                                          ltarget,
                                                                                          ETraceTypeQuery.TraceTypeQuery1,
                                                                                          DrawDebugType=EDrawDebugTrace.ForOneFrame,
                                                                                          bIgnoreSelf=True)
                dist = trace_length
                if is_hitting_something:
                    dist = hit_result.distance
                dist = dist / trace_length
                obs.append(dist)

        # TARGET ANGLE
        target_angle = self.get_target_angle()
        obs.append(target_angle)

        ##show target
        self.uobject.draw_debug_line(location,
                                     FVector(self.target_x, self.target_y, location.z),
                                     FLinearColor(0, 1, 0),
                                     0, 2)

        # DIST
        obs.append(min(self.get_target_dist() / 5000, 1))
        # CURRENT STEERING
        obs.append(self.actor.GetSteerAngle() * .5 + .5)
        state = np.array(obs)

        # env.render()##BIPEDAL
        # state = self.gstate ##BIPEDAL

        #############get action from policy###############
        action = self.policy.select_action(state)
        origaction = action

        if not self.actor.autodrive:
            # print("MANUAL")
            action[1] = self.actor.FIN
            action[0] = self.actor.RIN
            action[2] = 1 if self.actor.HBR else -1

        fwd = self.actor.get_actor_forward()
        rt = self.actor.get_actor_right()

        draw_debug_lines = False

        if draw_debug_lines:
            actionx = action[0] * 1000  # -.5
            actiony = action[1] * 1000  # -.5
            target_vec = location + fwd * actiony + rt * actionx
            self.uobject.draw_debug_line(location,
                                         target_vec,
                                         # FVector(location.x+action[0]*2000,location.y+action[1]*2000,location.z),
                                         FLinearColor(0, 0, 1),
                                         0,
                                         2)

        if self.actor.autodrive:
            random_normal = np.random.normal(0, exploration_noise, size=action_dim)
            action = action + random_normal
        action = action.clip([-1, -1, -1], [1, 1, 1])
        # action = action.clip(env.action_space.low, env.action_space.high)##BIPEDAL

        if draw_debug_lines:
            actionx = action[0] * 1000  # -.5
            actiony = action[1] * 1000  # -.5
            target_vec = location + fwd * actiony + rt * actionx
            self.uobject.draw_debug_line(location,
                                         target_vec,
                                         # FVector(location.x+action[0]*2000,location.y+action[1]*2000,location.z),
                                         FLinearColor(1, 0, 0),
                                         .05,
                                         2)

        ######### apply action ##########
        if self.actor.autodrive:
            self.actor.FIN = action[1]
            self.actor.RIN = action[0]
            self.actor.HBR = True if action[2] > .9 else False
            ###############apply action################
            self.VehicleMovement.SetThrottleInput(action[1])
            self.VehicleMovement.SetSteeringInput(action[0])
            self.VehicleMovement.SetHandbrakeInput(self.actor.HBR)

        ########## calc reward ###########
        new_dist = self.get_target_dist()
        diffd = self.last_dist - new_dist
        reward = diffd - 0.4
        # reward = self.greward##BIPEDAL
        self.last_dist = new_dist

        # self.last_done = self.gdone#False
        # if self.gdone: ##BIPEDAL
        #     env.reset()
        #     al, c1l, c2l, prl = self.policy.update(self.replay_buffer, 200, batch_size, gamma, polyak, policy_noise,noise_clip, policy_delay)

        done = 0
        if self.ep_frame > max_timesteps:
            done = 1

        angle_reward = abs(target_angle-.5)
        reward -= angle_reward

        # if (target_angle > .97 or target_angle < .03):
        #     print("DED")
        #     #done = 1
        #     reward -= 50

        if self.actor.Punish:
            self.actor.Punish = False
            reward -= 50
            print("PUNISH")
        if self.actor.Reward:
            self.actor.Reward = False
            reward += 50
            print("REWARD")
        if self.actor.Train:
            self.actor.Train = False
            al, c1l, c2l, prl = self.policy.update(self.replay_buffer, 200, batch_size, gamma, polyak, policy_noise,
                                                   noise_clip, policy_delay)
            print("Train")
        if self.actor.Reset:
            self.actor.Reset = False
            al, c1l, c2l, prl = self.policy.update(self.replay_buffer, 5000, batch_size, gamma, polyak, policy_noise,
                                                   noise_clip, policy_delay)
            done = 1
            print("Reset")

        self.ep_reward += reward

        if done:
            # self.actor.set_physics_linear_velocity(FVector(0,0,0))
            # self.actor.set_physics_angular_velocity(FVector(0,0,0))
            al, c1l, c2l, prl = self.policy.update(self.replay_buffer, self.ep_frame, batch_size, gamma, polyak, policy_noise,
                                                   noise_clip, policy_delay)
            ep_reward_avg = self.ep_reward/self.ep_frame
            self.writer.add_scalar('actor_loss',
                              al,
                              self.episode)
            self.writer.add_scalar('critic1_loss',
                              c1l,
                              self.episode)
            self.writer.add_scalar('ep_reward',
                              self.ep_reward,
                              self.episode)
            self.writer.add_scalar('ep_avg_reward',
                              ep_reward_avg,
                              self.episode)
            self.policy.save(directory, filename)
            if ep_reward_avg > self.ep_reward_avg_BEST:
                self.ep_reward_avg_BEST = ep_reward_avg
                self.policy.save(directory, filename+"_best")
            self.reset_ep()

        ####### record action ############
        if len(self.last_state):
            self.replay_buffer.add((self.last_state, self.last_action, reward, state, float(done)))
        self.last_state = state
        self.last_action = action
        # self.last_reward = reward

        self.frame += 1
        if self.frame > 2 and self.frame % 30 == 0:
            # self.actor.AutoDrive = True
            # al, c1l, c2l, prl = self.policy.update(self.replay_buffer, 5, batch_size, gamma, polyak, policy_noise,
            #                                        noise_clip, policy_delay)
            if self.frame % 60 == 0:
                eval = self.policy.eval_action(state, action)
                print(state)
                print(
                    "ep_frame{}:, action:{}, directaction:{}, eval:{}, reward: {}, memory:{}, ep:{}, interaction:{}".format(
                        self.ep_frame, action, origaction, eval, reward, self.replay_buffer.size, self.episode, self.frame))

        if new_dist < 500:
            self.gen_target()
            self.last_dist = self.get_target_dist()
            ue.log("NEW TARGET!!!!!")

        # next_state, reward, done, _ = env.step(action) ##BIPEDAL
        # self.gstate = next_state
        # self.gdone = done
        # self.greward = reward
