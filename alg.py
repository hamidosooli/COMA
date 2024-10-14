import torch
import torch.optim as optim
from network import DQN, RNN_AGENT
from tensordict import TensorDict

class COMA:
    def __init__(self, num_agents, num_states, num_observations, num_actions, device):
        self.device = device
        self.num_agents = num_agents

        self.num_observations = num_observations
        self.num_actions = num_actions

        self.num_states = num_states
        self.num_joint_actions = self.num_actions ** (self.num_agents-1)

        self.actors_input_size = self.num_observations + 1 + 1  # observations + action + agent_id
        self.critic_input_size = (self.num_agents-1) + self.num_states + self.num_observations + 1 + 1  # others_actions, state, observations, agent_id, action
        self.num_episodes = 1e5
        self.critic_update_steps = 200
        self.batch_size = 30
        self.memory_size = self.batch_size
        self.gamma = 0.99
        self.critic_lambda = 0.8
        self.learning_rate = 3e-4
        self.tau = 0.005
        self.agent_batch = int(self.batch_size / self.num_agents)
        self.memory = {'agent_id': torch.zeros(self.agent_batch, self.num_agents, 1),
                       'valid': torch.ones(self.agent_batch, self.num_agents, 1),
                       'state': torch.zeros(self.agent_batch, self.num_agents, self.num_states),
                       'action': torch.zeros(self.agent_batch, self.num_agents, 1),
                       'others_actions': torch.zeros(self.agent_batch, self.num_agents, self.num_agents-1),
                       'observation': torch.zeros(self.agent_batch, self.num_agents, self.num_observations),
                       'rewards': torch.zeros(self.agent_batch, self.num_agents, 1),
                       'policy': torch.zeros(self.agent_batch, self.num_agents, self.num_actions),
                       'action_values': torch.zeros(self.agent_batch, self.num_agents, self.num_actions),
                       'termination': torch.zeros(self.agent_batch, self.num_agents, 1)}
        self.team_memory = dict()
        self.memory = TensorDict(self.memory, batch_size=self.agent_batch, device=self.device)
        self.actors_net = RNN_AGENT(self.actors_input_size, self.num_actions, self.actors_input_size, self.device)
        self.critic_net = DQN(self.critic_input_size, self.num_joint_actions*self.num_actions, self.device)
        self.critic_target_net = DQN(self.critic_input_size, self.num_joint_actions*self.num_actions, self.device)
        self.critic_target_net.load_state_dict(self.critic_net.state_dict())

        self.actors_loss = 0.0
        self.critic_loss = 0.0

        self.actors_optimizer = optim.Adam(self.actors_net.parameters(), lr=self.learning_rate, betas=(0.9, 0.99))
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.learning_rate, betas=(0.9, 0.99))

    def memory_reset(self):
        self.memory = {'agent_id': torch.zeros(self.agent_batch, self.num_agents, 1),
                       'valid': torch.ones(self.agent_batch, self.num_agents, 1),
                       'state': torch.zeros(self.agent_batch, self.num_agents, self.num_states),
                       'action': torch.zeros(self.agent_batch, self.num_agents, 1),
                       'others_actions': torch.zeros(self.agent_batch, self.num_agents, self.num_agents-1),
                       'observation': torch.zeros(self.agent_batch, self.num_agents, self.num_observations),
                       'rewards': torch.zeros(self.agent_batch, self.num_agents, 1),
                       'policy': torch.zeros(self.agent_batch, self.num_agents, self.num_actions),
                       'action_values': torch.zeros(self.agent_batch, self.num_agents, self.num_actions),
                       'termination': torch.zeros(self.agent_batch, self.num_agents, 1)}

        self.memory = TensorDict(self.memory, batch_size=self.agent_batch, device=self.device)

        self.team_memory = dict()

    def optimize_actors(self):
        self.actors_optimizer.zero_grad()
        self.actors_loss.backward()
        actors_grad_norm = torch.nn.utils.clip_grad_norm_(self.actors_net.parameters(), 5)
        self.actors_optimizer.step()
        return actors_grad_norm

    def optimize_critic(self):
        self.critic_optimizer.zero_grad()
        self.critic_loss.backward(retain_graph=True)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 5)
        self.critic_optimizer.step()
        return critic_grad_norm

    def critic_target_soft_update(self):
        # Critic's target network soft update
        critic_target_net_state_dict = self.critic_target_net.state_dict()
        critic_net_dict = self.critic_net.state_dict()
        for key in critic_net_dict:
            critic_target_net_state_dict[key] = critic_net_dict[key] * self.tau + \
                                                critic_target_net_state_dict[key] * (1 - self.tau)

    def critic_target_hard_update(self, step):
        if step == self.critic_update_steps:
            # Critic's target network hard update
            self.critic_target_net.load_state_dict(self.critic_net.state_dict())

    def collate_episodes(self):
        self.team_memory = {'agent_id': self.memory['agent_id'].permute(1, 0, 2).contiguous().view(self.batch_size, 1),
                            'valid': self.memory['valid'].permute(1, 0, 2).contiguous().view(self.batch_size, 1),
                            'state': self.memory['state'].permute(1, 0, 2).contiguous().view(self.batch_size, self.num_states),
                            'action': self.memory['action'].permute(1, 0, 2).contiguous().view(self.batch_size, 1),
                            'others_actions': self.memory['others_actions'].permute(1, 0, 2).contiguous().view(self.batch_size, self.num_agents - 1),
                            'observation': self.memory['observation'].permute(1, 0, 2).contiguous().view(self.batch_size, self.num_observations),
                            'rewards': self.memory['rewards'].sum(dim=1).repeat_interleave(self.num_agents, dim=1).unsqueeze(-1).permute(1, 0, 2).contiguous().view(self.batch_size, 1),
                            'policy': self.memory['policy'].permute(1, 0, 2).contiguous().view(self.batch_size, self.num_actions),
                            'action_values': self.memory['action_values'].permute(1, 0, 2).contiguous().view(self.batch_size, self.num_actions),
                            'termination': self.memory['termination'].permute(1, 0, 2).contiguous().view(self.batch_size, 1)}

    def target_lambda_return(self, n_step=10):
        critic_inp = torch.cat([self.team_memory['others_actions'],
                                self.team_memory['state'],
                                self.team_memory['observation'],
                                self.team_memory['agent_id'],
                                self.team_memory['action']], -1)
        critic_action_value = self.critic_target_net(critic_inp)

        # combining the taken actions wih teammate actions
        actions = torch.cat([self.team_memory['others_actions'],
                             self.team_memory['action']], -1)
        # changing the actions from values to proper indices
        mutual_action_index = torch.sum(actions * torch.tensor([
            self.num_actions ** pow for pow in range(self.num_agents)], device=self.device), dim=1)
        # critic value for the taken action
        critic_value = torch.gather(critic_action_value, 1,
                                    index=mutual_action_index.unsqueeze(-1).type(torch.int64))
        # combining available actions wih teammate actions
        actions_ = self.team_memory['others_actions'].repeat_interleave(self.num_actions, dim=0)
        available_actions = torch.arange(self.num_actions, device=self.device).repeat(self.batch_size)
        available_actions_indices = torch.cat([actions_, available_actions.unsqueeze(-1)], -1)
        # changing the actions from values to proper indices
        mutual_action_index_ = torch.sum(available_actions_indices * torch.tensor([
            self.num_actions ** pow for pow in range(self.num_agents)], device=self.device), dim=1)
        # critic value for the available actions
        critic_value_ = torch.gather(critic_action_value, 1,
                                     index=mutual_action_index_.view(self.batch_size,
                                                                     self.num_actions).type(torch.int64))
        # calculation of the baseline
        baseline = critic_value_ * self.team_memory['policy']
        baseline = baseline.sum(dim=-1).detach()

        # calculation of the advantage
        advantage = critic_value - baseline.unsqueeze(-1)

        # n_step values includes a value for all states except terminal
        targets = torch.zeros_like(critic_value[:-1])
        # see the n-step TD on Sutton & Barto pp.144
        for t in range(self.batch_size-1):
            n_step_return = 0.0
            lambda_coeff = 1
            for n in range(n_step+1):
                tau = t + n
                if tau >= self.batch_size-1:
                    break
                elif n == n_step:
                    n_step_return += lambda_coeff * (self.gamma**n) * critic_value[tau] * self.team_memory['valid'][tau]
                elif tau == self.batch_size-2:
                    n_step_return += lambda_coeff * (self.gamma**n) * self.team_memory['rewards'][tau] * self.team_memory['valid'][tau]
                    n_step_return += lambda_coeff * (self.gamma**(n+1)) * critic_value[tau+1]
                else:
                    n_step_return += lambda_coeff * (self.gamma**n) * self.team_memory['rewards'][tau] * self.team_memory['valid'][tau]
                # lambda_coeff *= self.critic_lambda
            targets[t] = n_step_return
            # targets[t] = (1 - self.critic_lambda) * n_step_return
        return targets, advantage



