import wandb
from alg import COMA
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from pettingzoo.mpe import simple_tag_v3, simple_spread_v3  # predator-prey, cooperation

# if GPU is supposed to be used
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'Training started on {device}')
AGENT_BATCH_SIZE = 10
env = simple_spread_v3.env(max_cycles=AGENT_BATCH_SIZE, continuous_actions=False, render_mode=None)
env.reset(seed=42)
WANDB = True
if WANDB:
    wandb.init(project="Custom_COMA")

num_agents = env.num_agents
num_observations = env.env.observation_spaces['agent_0'].shape[0]
num_actions = env.env.action_space('agent_0').n
num_states = env.env.state_space.shape[0]
alg = COMA(num_agents, num_states, num_observations, num_actions, device)
if WANDB:
    wandb.watch([alg.critic_target_net, alg.critic_net, alg.actors_net])

for eps in tqdm(range(int(alg.num_episodes))):

    env.reset()
    alg.memory_reset()  # empty buffer
    '''
    Collect Data
    '''
    state = env.state()
    t = 0
    hidden_state = alg.actors_net.init_hidden().to(device)
    termination = False
    truncation = False
    # initial random action
    action = torch.randint(0, alg.num_actions, (1,), device=device).item()
    while not termination and t < alg.agent_batch:

        for agent in env.agent_iter(max_iter=alg.num_agents):
            # update the agent's local observation and get the reward
            observation, reward, termination, truncation, info = env.last()
            # change the observation vector into a torch tensor
            observation = torch.tensor(observation, device=device, requires_grad=False)
            # update the central state
            state = torch.tensor(env.state(), device=device, requires_grad=False)
            if termination or truncation:
                action = None
            else:
                # change the action vector into a torch tensor
                action = torch.tensor(action, device=device, requires_grad=False).unsqueeze(-1)
                # get the agent's id
                agent_id = torch.tensor([int(agent[-1])], device=device, requires_grad=False)

                # make a vector out of the agent's local information (o, a, i)
                actor_inp = torch.cat([observation, action, agent_id], -1)
                # get the q and update the hidden state for the actor
                q, hidden_state = alg.actors_net(actor_inp.unsqueeze(0), hidden_state[agent_id])

                # softmax over the calculated action values to get the policy
                policy = F.softmax(q, -1)
                # argmax over the policy values to calculate the taken action
                action = Categorical(policy).sample().item()

            # apply the action into the environment
            env.step(action)

            # get the vector of the teammates actions and turn it into a one-hot vector
            actions = [act if act is not None else alg.num_actions for act in env.env.current_actions]
            others_actions = actions[:agent_id.item()] + actions[agent_id.item()+1:]
            others_actions = torch.tensor(others_actions, dtype=torch.float32, device=device, requires_grad=False)

            # add episode to buffer
            alg.memory['agent_id'][t, agent_id, 0] = agent_id.type(torch.float)
            alg.memory['valid'][t, agent_id, 0] = torch.tensor([1. if alg.num_actions not in actions else 0.],
                                                               requires_grad=False)
            alg.memory['state'][t, agent_id, :] = state
            alg.memory['action'][t, agent_id, 0] = alg.num_actions if action is None else action
            alg.memory['others_actions'][t, agent_id, :] = others_actions
            alg.memory['observation'][t, agent_id, :] = observation
            alg.memory['rewards'][t, agent_id, 0] = reward
            alg.memory['policy'][t, agent_id, :] = policy
            alg.memory['action_values'][t, agent_id, :] = q
            alg.memory['termination'][t, agent_id, 0] = truncation or termination
        t += 1
    ''''
    Collate episodes in buffer
    '''
    alg.collate_episodes()
    ''''
    Calculate TD(λ) targets
    '''
    targets, advantage = alg.target_lambda_return()
    targets = (targets-targets.mean()) / torch.sqrt(targets.var())
    '''
    Train critic
    '''
    critic_inp = torch.cat([alg.team_memory['others_actions'],
                            alg.team_memory['state'],
                            alg.team_memory['observation'],
                            alg.team_memory['agent_id'],
                            alg.team_memory['action']], -1)
    critic_action_value = alg.critic_net(critic_inp)
    # combining the taken actions wih teammate actions
    actions = torch.cat([alg.team_memory['others_actions'],
                         alg.team_memory['action']], -1)
    # changing the actions from values to proper indices
    mutual_action_index = torch.sum(actions * torch.tensor([
        alg.num_actions ** pow for pow in range(alg.num_agents)], device=alg.device), dim=1)
    # critic value for the taken action
    critic_value = torch.gather(critic_action_value, 1,
                                index=mutual_action_index.unsqueeze(-1).type(torch.int64))
    td_error = (critic_value[:-1] - targets) * alg.team_memory['valid'][:-1]
    alg.critic_loss = (td_error**2).sum() / alg.team_memory['valid'][:-1].sum()
    critic_grad_norm = alg.optimize_critic()

    alg.critic_target_soft_update()
    # alg.critic_target_hard_update(step)
    '''
    Train actors
    '''
    # getting the chosen policy for the actors
    policy_taken = torch.gather(alg.team_memory['policy'], 1, index=alg.team_memory['action'].type(torch.int64))

    # masking the policy for invalid action time steps
    policy_taken[alg.team_memory['valid'] == 0] = 1.0
    log_policy_taken = torch.log(policy_taken)

    actors_grad = (log_policy_taken * advantage).sum()
    # average over the valid losses
    alg.actors_loss = -actors_grad / alg.team_memory['valid'].sum()
    actors_grad_norm = alg.optimize_actors()

    if WANDB:
        wandb.log({"targets_mean": torch.mean(targets),
                   "targets_std": torch.std(targets),

                   "critic_grad_norm": critic_grad_norm,
                   "advantage_mean": torch.mean(advantage),
                   "critic_loss": alg.critic_loss,

                   "actors_grad_norm": actors_grad_norm,
                   "actors_loss": alg.actors_loss,

                   "episode": eps})
env.close()
