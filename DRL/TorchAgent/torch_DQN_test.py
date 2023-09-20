
import math

import gym

from torch_method.torch_DQN import DQN_generate

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * \
                                     math.exp(-1. * frame_idx / epsilon_decay)
env_id = "CartPole-v0"
env = gym.make(env_id)

num_frames = 10000
batch_size = 32
gamma      = 0.99

losses = []
all_rewards = []
episode_reward = 0

state = env.reset()
DQN_test = DQN_generate(env.observation_space.shape[0], env.action_space.n,1000)

for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    action = DQN_test.current_model.act(state, epsilon)

    next_state, reward, done, _ = env.step(int(action))
    DQN_test.replay_buffer.push(state, action, reward, next_state, done)

    state = next_state
    episode_reward += reward

    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0

    if len(DQN_test.replay_buffer) > batch_size:
        loss = DQN_test.compute_td_loss(batch_size)
        losses.append(loss.data)

    if frame_idx % 200 == 0:
        DQN_test.plot(frame_idx, all_rewards)

    if frame_idx % 100 == 0:
        DQN_test.update_target()
