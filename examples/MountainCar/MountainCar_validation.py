import gym
from gym.core import Wrapper
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

use_dessca = False


with open("ValidationInits.json", 'r') as json_file:
    data = json.load(json_file)
    initial_states = np.copy(data["initial_states"])


class mountain_car_reset_wrapper(Wrapper):

    def __init__(self, environment, try_nb):
        super().__init__(environment)

        self.try_nb = try_nb
        self.state_saver = None
        self._episode_reward_list = []

        self._init_state_iterator = 0

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        self.state_saver = np.append(self.state_saver, np.reshape(self.env.env.state, (2, 1)), axis=1)
        self._reward_list.append(reward)
        self.k += 1

        if self.k == 200:
            done = True

        norm_state = (self.env.env.state - self.env.env.low_state) / (self.env.env.high_state - self.env.env.low_state)
        norm_state = norm_state * 2 - 1

        return norm_state, reward, done, _

    def reset(self, **kwargs):

        if hasattr(self, "_reward_list"):
            self._episode_reward_list.append(self._reward_list)

        self._reward_list = []
        self.k = 0

        self.env.reset()
        self.env.env.state = initial_states[:, self._init_state_iterator]
        self._init_state_iterator += 1

        if self.state_saver is None:
            self.state_saver = np.reshape(np.copy(self.env.env.state), (2, 1))
        else:
            self.state_saver = np.append(self.state_saver, np.reshape(self.env.env.state, (2, 1)), axis=1)

        norm_state = (self.env.env.state - self.env.env.low_state) / (self.env.env.high_state - self.env.env.low_state)
        norm_state = norm_state * 2 - 1

        return norm_state

    def close(self):
        if hasattr(self, "_reward_list"):
            self._episode_reward_list.append(self._reward_list)
        self.env.close()

        save_dict = {
            "state_history": self.state_saver.tolist(),
            "reward_history": self._episode_reward_list
        }
        if use_dessca:
            experiment = "Dessca"
        else:
            experiment = "Uniform"

        with open('./' + experiment + "/" + experiment + '_Validation_' + str(self.try_nb) + '.json', 'w') as json_file:
            json.dump(save_dict, json_file)

nb_agents = 50
for i in range(nb_agents):

    env = gym.make('MountainCarContinuous-v0')
    env = mountain_car_reset_wrapper(env, i)

    nb_actions = env.action_space.shape[0]
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(16, activation='relu'))
    actor.add(Dense(16, activation='relu'))
    actor.add(Dense(nb_actions, activation='linear',))
    actor.add(Lambda(lambda x: tf.clip_by_value(x, -1, 1)))


    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(1, activation='linear')(x)
    critic = Model(inputs=(action_input, observation_input), outputs=x)

    random_process = OrnsteinUhlenbeckProcess(
        theta=0.0,
        mu=0.0,
        sigma=0.0,
        dt=1,
        sigma_min=0.00,
        n_steps_annealing=20000,
        size=nb_actions
    )

    memory = SequentialMemory(
        limit=5000,
        window_length=1
    )

    agent = DDPGAgent(
        # Pass the previously defined characteristics
        nb_actions=nb_actions,
        actor=actor,
        critic=critic,
        critic_action_input=action_input,
        memory=memory,
        random_process=random_process,
        # Define the overall training parameters
        nb_steps_warmup_actor=64,
        nb_steps_warmup_critic=64,
        target_model_update=100,
        gamma=0.9,
        batch_size=16,
        memory_interval=1
    )

    agent.compile([Adam(lr=1e-5), Adam(lr=1e-4)])

    if use_dessca:
        experiment = "Dessca"
    else:
        experiment = "Uniform"

    agent.load_weights(filepath="./" + experiment + "/" + experiment + "_weights_" + str(i) + ".hdf5")

    history = agent.test(env,
                         nb_episodes=1000,
                         action_repetition=1,
                         verbose=0,
                         visualize=False,
                         nb_max_episode_steps=500)

    print(f"Done with Agent {i} / {nb_agents-1}")
    env.close()
