import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 400)
        self.l3 = nn.Linear(400, 300)
        self.l4 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        a = torch.tanh(self.l4(a)) * self.max_action
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 400)
        self.l3 = nn.Linear(400, 300)
        self.l4 = nn.Linear(300, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q = F.relu(self.l1(state_action))
        q = F.relu(self.l2(q))
        q = F.relu(self.l3(q))
        q = self.l4(q)
        return q


# class Predictor(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Predictor, self).__init__()
#
#         self.l1 = nn.Linear(state_dim + action_dim, 400)
#         self.l2 = nn.Linear(400, 400)
#         self.l3 = nn.Linear(400, 400)
#         self.l4 = nn.Linear(400, state_dim)
#
#     def forward(self, state, action):
#         state_action = torch.cat([state, action], 1)
#
#         s = F.relu(self.l1(state_action))
#         s = F.relu(self.l2(s))
#         s = F.relu(self.l3(s))
#         s = self.l4(s)
#         return s


class TD3:
    def __init__(self, lr, state_dim, action_dim, max_action):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr*10)

        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr*10)

        # self.guesser = Predictor(state_dim, action_dim).to(device)
        # self.guesser_target = Predictor(state_dim, action_dim).to(device)
        # self.guesser_target.load_state_dict(self.guesser.state_dict())
        # self.guesser_optimizer = optim.Adam(self.guesser.parameters(), lr=lr)

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def guess_state(self, state, action):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(device)
        return self.guesser(state, action).cpu().data.numpy().flatten()

    def eval_action(self, state, action):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(device)
        current_Q1 = self.critic_1(state, action)
        # current_Q2 = self.critic_2(state, action)
        # current_Q = torch.min(current_Q1, current_Q2)
        # return current_Q.cpu().data.numpy().flatten()[0]
        return current_Q1.cpu().data.numpy().flatten()[0]
        # print(current_Q1.__class__)

    def update(self, replay_buffer, n_iter, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay):
        critic_only = False
        al = 0
        c1l = 0
        c2l = 0
        prl = 0
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action_, reward, next_state, done = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action_).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size, 1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).reshape((batch_size, 1)).to(device)

            # # predict next state
            # # cstate = torch.FloatTensor(state.reshape(1, -1)).to(device)
            # # caction = torch.FloatTensor(action.reshape(1, -1)).to(device)
            # pred_state = self.guesser(state, action)
            # # pred_state = self.guess_state()
            #
            # loss_pred_state = F.mse_loss(pred_state, next_state)
            # prl += loss_pred_state.cpu().data.numpy().flatten()[0]
            # self.guesser_optimizer.zero_grad()
            # loss_pred_state.backward()
            # self.guesser_optimizer.step()

            # Select next action according to target policy:
            noise = torch.FloatTensor(action_).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            c1l += loss_Q1.cpu().data.numpy().flatten()[0]
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            c2l += loss_Q2.cpu().data.numpy().flatten()[0]
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()

            # Delayed policy updates:
            if i % policy_delay == 0:
                if not critic_only:
                    # Compute actor loss:
                    actor_loss = -self.critic_1(state, self.actor(state)).mean()

                    al += actor_loss.cpu().data.numpy().flatten()[0]

                    # Optimize the actor
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # Polyak averaging update:
                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_((polyak * target_param.data) + ((1 - polyak) * param.data))

                    for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                        target_param.data.copy_((polyak * target_param.data) + ((1 - polyak) * param.data))

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_((polyak * target_param.data) + ((1 - polyak) * param.data))
        return al / n_iter, c1l / n_iter, c2l / n_iter, prl / n_iter

    def blend_from_another(self, other, polyak):
        #used to blend in params from another policy
        #e.g. - a central learner

        for param, target_param in zip(self.actor.parameters(), other.actor.parameters()):
            target_param.data.copy_((polyak * target_param.data) + ((1 - polyak) * param.data))

        for param, target_param in zip(self.critic_1.parameters(), other.critic_1.parameters()):
            target_param.data.copy_((polyak * target_param.data) + ((1 - polyak) * param.data))

        for param, target_param in zip(self.critic_2.parameters(), other.critic_2.parameters()):
            target_param.data.copy_((polyak * target_param.data) + ((1 - polyak) * param.data))

        for param, target_param in zip(self.actor_target.parameters(), other.actor_target.parameters()):
            target_param.data.copy_((polyak * target_param.data) + ((1 - polyak) * param.data))

        for param, target_param in zip(self.critic_1_target.parameters(), other.critic_1_target.parameters()):
            target_param.data.copy_((polyak * target_param.data) + ((1 - polyak) * param.data))

        for param, target_param in zip(self.critic_2_target.parameters(), other.critic_2_target.parameters()):
            target_param.data.copy_((polyak * target_param.data) + ((1 - polyak) * param.data))

    def save(self, directory, name):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, name))

        torch.save(self.critic_1.state_dict(), '%s/%s_crtic_1.pth' % (directory, name))
        torch.save(self.critic_1_target.state_dict(), '%s/%s_critic_1_target.pth' % (directory, name))

        torch.save(self.critic_2.state_dict(), '%s/%s_crtic_2.pth' % (directory, name))
        torch.save(self.critic_2_target.state_dict(), '%s/%s_critic_2_target.pth' % (directory, name))

        # torch.save(self.guesser.state_dict(), '%s/%s_guesser.pth' % (directory, name))
        # torch.save(self.guesser_target.state_dict(), '%s/%s_guesser_target.pth' % (directory, name))

    def load(self, directory, name):
        self.actor.load_state_dict(
            torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(
            torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))

        self.critic_1.load_state_dict(
            torch.load('%s/%s_crtic_1.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_1_target.load_state_dict(
            torch.load('%s/%s_critic_1_target.pth' % (directory, name), map_location=lambda storage, loc: storage))

        self.critic_2.load_state_dict(
            torch.load('%s/%s_crtic_2.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_2_target.load_state_dict(
            torch.load('%s/%s_critic_2_target.pth' % (directory, name), map_location=lambda storage, loc: storage))

        # self.guesser.load_state_dict(
        #     torch.load('%s/%s_guesser.pth' % (directory, name), map_location=lambda storage, loc: storage))
        # self.guesser_target.load_state_dict(
        #     torch.load('%s/%s_guesser_target.pth' % (directory, name), map_location=lambda storage, loc: storage))

    def load_actor(self, directory, name):
        self.actor.load_state_dict(
            torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(
            torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))





