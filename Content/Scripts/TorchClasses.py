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

from unreal_engine import FVector, FRotator, FTransform, FHitResult, FLinearColor
from unreal_engine.classes import ActorComponent, ForceFeedbackEffect, KismetSystemLibrary, WidgetBlueprintLibrary
from unreal_engine.enums import EInputEvent, ETraceTypeQuery, EDrawDebugTrace

trace_num = 0
state_dim = trace_num+3
action_dim = 2
max_action = 1.0

max_power = 60000
trace_length = 100*20

gamma = 0.99  # discount for future rewards
batch_size = 25  # num of transitions sampled from replay buffer
lr = 0.001
exploration_noise_max = 1.0
exploration_noise = 0.9
polyak = 0.95  # target policy update parameter (1-tau)
policy_noise = 0.2  # target policy smoothing noise
noise_clip = 0.5
policy_delay = 2  # delayed policy updates parameter

class TorchActor:

    # this is called on game start
    def begin_play(self):
        self.replay_buffer = ReplayBuffer(max_size=5000)
        ue.log('Begin Play on TorchActor class')
        ue.log(torch.cuda.is_available())
        ue.log(dir(self.uobject))

        self.policy = TD3(lr, state_dim, action_dim, max_action)
        self.components = self.uobject.get_actor_components()
        self.mainbody = self.components[0]
        #ue.log("---COMONETNTS---")
        #ue.log(components)
        self.gen_target()

        self.last_dist = self.get_target_dist()
        self.last_state = []
        self.last_reward = 0
        self.last_action = None
        self.last_done = False
        self.frame = 0 # int(random.random()*100)
        self.start_pos = self.uobject.get_actor_location()
    def gen_target(self):
        target_angle = random.random()*math.pi*2.0
        target_dist = random.random()*100*20
        self.target_x = math.cos(target_angle)*target_dist
        self.target_y = math.sin(target_angle)*target_dist
    def get_target_angle(self):
        location = self.uobject.get_actor_location()
        xd = location.x-self.target_x
        yd = location.y-self.target_y
        rangle = math.atan2(yd,xd)
        nangle = rangle+math.pi
        nangle = nangle/(math.pi*2)
        return nangle
    def get_target_dist(self):
        location = self.uobject.get_actor_location()
        xd = location.x-self.target_x
        yd = location.y-self.target_y
        return math.sqrt(xd*xd+yd*yd)


    # this is called at every 'tick'
    def tick(self, delta_time):
        #############get observation#############

        #VELOCITY
        vel = self.mainbody.get_physics_linear_velocity()
        #ue.log(vel)
        obs = [vel.x/1000+.5, vel.y/1000+.5]

        #LIDAR
        location = self.uobject.get_actor_location()
        if(trace_num):
            for trace_id in range(trace_num):
                angle = trace_id/trace_num*math.pi*2.0
                ltarget = location + FVector(math.cos(angle)*trace_length, math.sin(angle)*trace_length, 0)
                is_hitting_something, _, hit_result = KismetSystemLibrary.LineTraceSingle(self.uobject,
                                                                                           location,
                                                                                           ltarget,
                                                                                           ETraceTypeQuery.TraceTypeQuery1,
                                                                                           #DrawDebugType=EDrawDebugTrace.ForOneFrame,
                                                                                           bIgnoreSelf=True)
                dist = trace_length
                if is_hitting_something:
                    dist = hit_result.distance
                dist = dist/trace_length
                obs.append(dist)
                # ue.log("---FRAME---")
                # ue.log(is_hitting_something)
                # ue.log(hit_result)
                # ue.log(hit_result.distance)
                # #ue.log(x)
                # ue.log("---end---")

        target_angle = self.get_target_angle()
        obs.append(target_angle)
        #TARGET ANGLE
        #ue.log(len(obs))
        # KismetSystemLibrary.LineTraceSingle(self.uobject,
        #                                     location,
        #                                     FVector(self.target_x,self.target_y,location.z),
        #                                     ETraceTypeQuery.TraceTypeQuery1,
        #                                     DrawDebugType=EDrawDebugTrace.ForOneFrame,
        #                                     bIgnoreSelf=True)

        ##show target
        self.uobject.draw_debug_line(location,
                                    FVector(self.target_x,self.target_y,location.z),
                                     FLinearColor(0,1,0))

        # increase Z honouring delta_time
        # location.z += 100 * delta_time
        # set new location

        #self.uobject.set_actor_location(location)


        #############get action from poliucy###############
        state = np.array(obs)
        #print(state)
        action = self.policy.select_action(state)
        random_normal = np.random.normal(0, exploration_noise, size=action_dim)

        self.uobject.draw_debug_line(location,
                                    FVector(location.x+action[0]*2000,location.y+action[1]*2000,location.z),
                                     FLinearColor(0,0,1),
                                     .1,
                                     3)

        action = action+random_normal
        action = action.clip([-1,-1], [1,1])


        self.uobject.draw_debug_line(location,
                                    FVector(location.x+action[0]*2000,location.y+action[1]*2000,location.z),
                                     FLinearColor(1,0,0),
                                     .1,
                                     3)
        #ue.log(action)

        actionx = action[0]#-.5
        actiony = action[1]#-.5

        ###############apply action################
        self.mainbody.add_force(FVector(actionx*max_power,actiony*max_power,0))
        #ue.log(location)

        done = 0
        new_dist = self.get_target_dist()

        diffd = self.last_dist-new_dist
        reward = max(diffd,0)
        reward = min(reward,2)
        #reward *= 3
        reward -=0.2
        #reward*=-1
        #ue.log(reward)


        self.last_dist = new_dist


        self.last_done = False
        if self.frame%1000 == 0:
            self.last_done = True
            self.uobject.set_actor_location(self.start_pos)
            self.mainbody.set_physics_linear_velocity(FVector(0,0,0))
            self.mainbody.set_physics_angular_velocity(FVector(0,0,0))
            self.gen_target()
            self.last_dist = self.get_target_dist()

        #######record action
        if len(self.last_state):
            self.replay_buffer.add((self.last_state, self.last_action, reward, state, float(self.last_done)))
        self.last_state = state
        self.last_action = action
        self.last_reward = reward



        self.frame+=1
        if self.frame>100 and self.frame%30 == 0:
            al,c1l,c2l, prl = self.policy.update(self.replay_buffer, 4, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
            if self.frame%60 == 0:
                eval = self.policy.eval_action(state, action)
                print("state:{}, action:{}, loss:{}, eval:{}, reward: {}, memory:{}".format(state,action,al, eval, reward, self.replay_buffer.size))


        if new_dist < 500:
            self.gen_target()
            self.last_dist = self.get_target_dist()
            ue.log("NEW TARGET!!!!!")


