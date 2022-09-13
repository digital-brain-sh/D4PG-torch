import os
import copy
import queue
import shutil
import time
import torch
import multiprocessing as mp

from time import sleep
from collections import deque
from datetime import datetime
from multiprocessing import set_start_method

import torch.multiprocessing as torch_mp

try:
    set_start_method("spawn")
except RuntimeError:
    pass

from cdma.envs.utils import create_env_wrapper

from .d4pg import LearnerD4PG
from .networks import PolicyNetwork
from .replay_buffer import create_replay_buffer
from .utils import OUNoise, make_gif, empty_torch_queue, Logger


class Agent(object):
    def __init__(
        self,
        config,
        policy,
        global_episode,
        n_agent=0,
        agent_type="exploration",
        log_dir="",
    ):
        print(f"Initializing agent {n_agent}...")
        self.config = config
        self.n_agent = n_agent
        self.agent_type = agent_type
        self.max_steps = config["max_ep_length"]
        self.num_episode_save = config["num_episode_save"]
        self.global_episode = global_episode
        self.local_episode = 0
        self.log_dir = log_dir

        # Create environment
        self.env_wrapper = create_env_wrapper(config)
        self.ou_noise = OUNoise(
            dim=config["action_dim"],
            low=config["action_low"],
            high=config["action_high"],
        )
        self.ou_noise.reset()

        self.actor = policy
        print("Agent ", n_agent, self.actor.device)

        # Logger
        log_path = f"{log_dir}/agent-{agent_type}-{n_agent}"
        self.logger = Logger(log_path)

    def update_actor_learner(self, learner_w_queue, training_on):
        """Update local actor to the actor from learner."""
        if not training_on.value:
            return
        try:
            source = learner_w_queue.get_nowait()
        except:
            return
        target = self.actor
        for target_param, source_param in zip(target.parameters(), source):
            w = torch.tensor(source_param).float()
            target_param.data.copy_(w)

    def run(self, training_on, replay_queue, learner_w_queue, update_step):
        # Initialise deque buffer to store experiences for N-step returns
        self.exp_buffer = deque()

        best_reward = -float("inf")
        rewards = []
        while training_on.value:
            episode_reward = 0
            num_steps = 0
            self.local_episode += 1
            self.global_episode.value += 1
            self.exp_buffer.clear()

            if self.local_episode % 100 == 0:
                print(f"Agent: {self.n_agent}  episode {self.local_episode}")

            ep_start_time = time.time()
            state = self.env_wrapper.reset()
            self.ou_noise.reset()
            done = False
            while not done:
                action = self.actor.get_action(state)
                if self.agent_type == "exploration":
                    action = self.ou_noise.get_action(action, num_steps)
                    action = action.squeeze(0)
                else:
                    action = action.detach().cpu().numpy().flatten()
                next_state, reward, done = self.env_wrapper.step(action)
                num_steps += 1

                if num_steps == self.max_steps:
                    done = False
                episode_reward += reward

                state = self.env_wrapper.normalise_state(state)
                reward = self.env_wrapper.normalise_reward(reward)

                self.exp_buffer.append((state, action, reward))

                # We need at least N steps in the experience buffer before we can compute Bellman
                # rewards and add an N-step experience to replay memory
                if len(self.exp_buffer) >= self.config["n_step_returns"]:
                    state_0, action_0, reward_0 = self.exp_buffer.popleft()
                    discounted_reward = reward_0
                    gamma = self.config["discount_rate"]
                    for (_, _, r_i) in self.exp_buffer:
                        discounted_reward += r_i * gamma
                        gamma *= self.config["discount_rate"]
                    # We want to fill buffer only with form explorator
                    if self.agent_type == "exploration":
                        try:
                            replay_queue.put_nowait(
                                [
                                    state_0,
                                    action_0,
                                    discounted_reward,
                                    next_state,
                                    done,
                                    gamma,
                                ]
                            )
                        except:
                            pass

                state = next_state

                if done or num_steps == self.max_steps:
                    # add rest of experiences remaining in buffer
                    while len(self.exp_buffer) != 0:
                        state_0, action_0, reward_0 = self.exp_buffer.popleft()
                        discounted_reward = reward_0
                        gamma = self.config["discount_rate"]
                        for (_, _, r_i) in self.exp_buffer:
                            discounted_reward += r_i * gamma
                            gamma *= self.config["discount_rate"]
                        if self.agent_type == "exploration":
                            try:
                                replay_queue.put_nowait(
                                    [
                                        state_0,
                                        action_0,
                                        discounted_reward,
                                        next_state,
                                        done,
                                        gamma,
                                    ]
                                )
                            except:
                                pass
                    break

            # Log metrics
            step = update_step.value
            self.logger.scalar_summary(
                f"agent_{self.agent_type}/reward", episode_reward, step
            )
            self.logger.scalar_summary(
                f"agent_{self.agent_type}/episode_timing",
                time.time() - ep_start_time,
                step,
            )

            # Saving agent
            reward_outperformed = (
                episode_reward - best_reward > self.config["save_reward_threshold"]
            )
            time_to_save = self.local_episode % self.num_episode_save == 0
            if self.agent_type == "exploitation" and (
                time_to_save or reward_outperformed
            ):
                if episode_reward > best_reward:
                    best_reward = episode_reward
                self.save(f"local_episode_{self.local_episode}_reward_{best_reward:4f}")

            rewards.append(episode_reward)
            if (
                self.agent_type == "exploration"
                and self.local_episode % self.config["update_agent_ep"] == 0
            ):
                self.update_actor_learner(learner_w_queue, training_on)

        empty_torch_queue(replay_queue)
        print(f"Agent {self.n_agent} done.")

    def save(self, checkpoint_name):
        process_dir = f"{self.log_dir}/agent_{self.n_agent}"
        if not os.path.exists(process_dir):
            os.makedirs(process_dir)
        model_fn = f"{process_dir}/{checkpoint_name}.pt"
        torch.save(self.actor, model_fn)

    def save_replay_gif(self, output_dir_name):
        import matplotlib.pyplot as plt

        dir_name = output_dir_name
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        state = self.env_wrapper.reset()
        for step in range(self.max_steps):
            action = self.actor.get_action(state)
            action = action.cpu().detach().numpy()
            next_state, reward, done = self.env_wrapper.step(action)
            img = self.env_wrapper.render()
            plt.imsave(fname=f"{dir_name}/{step}.png", arr=img)
            state = next_state
            if done:
                break

        fn = f"{self.config['env']}-{self.config['model']}-{step}.gif"
        make_gif(dir_name, f"{self.log_dir}/{fn}")
        shutil.rmtree(dir_name, ignore_errors=False, onerror=None)
        print("fig saved to ", f"{self.log_dir}/{fn}")


def sampler_worker(
    config,
    replay_queue,
    batch_queue,
    replay_priorities_queue,
    training_on,
    global_episode,
    update_step,
    log_dir="",
):
    """
    Function that transfers replay to the buffer and batches from buffer to the queue.

    Args:
        config:
        replay_queue:
        batch_queue:
        training_on:
        global_episode:
        log_dir:
    """
    batch_size = config["batch_size"]
    logger = Logger(f"{log_dir}/data_struct")

    # Create replay buffer
    replay_buffer = create_replay_buffer(config)

    while training_on.value:
        # (1) Transfer replays to global buffer
        n = replay_queue.qsize()
        for _ in range(n):
            replay = replay_queue.get()
            replay_buffer.add(*replay)

        # (2) Transfer batch of replay from buffer to the batch_queue
        if len(replay_buffer) < batch_size:
            continue

        try:
            inds, weights = replay_priorities_queue.get_nowait()
            replay_buffer.update_priorities(inds, weights)
        except queue.Empty:
            pass

        try:
            batch = replay_buffer.sample(batch_size)
            batch_queue.put_nowait(batch)
        except:
            sleep(0.1)
            continue

        # Log data structures sizes
        step = update_step.value
        logger.scalar_summary("data_struct/global_episode", global_episode.value, step)
        logger.scalar_summary("data_struct/replay_queue", replay_queue.qsize(), step)
        logger.scalar_summary("data_struct/batch_queue", batch_queue.qsize(), step)
        logger.scalar_summary("data_struct/replay_buffer", len(replay_buffer), step)

    if config["save_buffer_on_disk"]:
        replay_buffer.dump(config["results_path"])

    empty_torch_queue(batch_queue)
    print("Stop sampler worker.")


def learner_worker(
    config,
    training_on,
    policy,
    target_policy_net,
    learner_w_queue,
    replay_priority_queue,
    batch_queue,
    update_step,
    experiment_dir,
):
    learner = LearnerD4PG(
        config, policy, target_policy_net, learner_w_queue, log_dir=experiment_dir
    )
    learner.run(training_on, batch_queue, replay_priority_queue, update_step)


def agent_worker(
    config,
    policy,
    learner_w_queue,
    global_episode,
    i,
    agent_type,
    experiment_dir,
    training_on,
    replay_queue,
    update_step,
):
    agent = Agent(
        config,
        policy=policy,
        global_episode=global_episode,
        n_agent=i,
        agent_type=agent_type,
        log_dir=experiment_dir,
    )
    agent.run(training_on, replay_queue, learner_w_queue, update_step)


class Engine(object):
    def __init__(self, config):
        self.config = config

    def train(self):
        config = self.config

        batch_queue_size = config["batch_queue_size"]
        n_agents = config["num_agents"]

        # Create directory for experiment
        experiment_dir = f"{config['results_path']}/{config['env']}-{config['model']}-{datetime.now():%Y-%m-%d_%H:%M:%S}"
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        # Data structures
        processes = []
        replay_queue = mp.Queue(maxsize=config["replay_queue_size"])
        training_on = mp.Value("i", 1)
        update_step = mp.Value("i", 0)
        global_episode = mp.Value("i", 0)
        learner_w_queue = torch_mp.Queue(maxsize=n_agents)
        replay_priorities_queue = mp.Queue(maxsize=config["replay_queue_size"])

        # Data sampler
        batch_queue = mp.Queue(maxsize=batch_queue_size)
        p = torch_mp.Process(
            target=sampler_worker,
            args=(
                config,
                replay_queue,
                batch_queue,
                replay_priorities_queue,
                training_on,
                global_episode,
                update_step,
                experiment_dir,
            ),
        )
        processes.append(p)

        # Learner (neural net training process)
        target_policy_net = PolicyNetwork(
            config["state_dim"],
            config["action_dim"],
            config["dense_size"],
            device=config["device"],
        )
        policy_net = copy.deepcopy(target_policy_net)
        policy_net_cpu = PolicyNetwork(
            config["state_dim"],
            config["action_dim"],
            config["dense_size"],
            device=config["agent_device"],
        )
        target_policy_net.share_memory()

        p = torch_mp.Process(
            target=learner_worker,
            args=(
                config,
                training_on,
                policy_net,
                target_policy_net,
                learner_w_queue,
                replay_priorities_queue,
                batch_queue,
                update_step,
                experiment_dir,
            ),
        )
        processes.append(p)

        # Single agent for exploitation
        p = torch_mp.Process(
            target=agent_worker,
            args=(
                config,
                target_policy_net,
                None,
                global_episode,
                0,
                "exploitation",
                experiment_dir,
                training_on,
                replay_queue,
                update_step,
            ),
        )
        processes.append(p)

        # Agents (exploration processes)
        for i in range(n_agents):
            p = torch_mp.Process(
                target=agent_worker,
                args=(
                    config,
                    copy.deepcopy(policy_net_cpu),
                    learner_w_queue,
                    global_episode,
                    i,
                    "exploration",
                    experiment_dir,
                    training_on,
                    replay_queue,
                    update_step,
                ),
            )
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        print("End.")
