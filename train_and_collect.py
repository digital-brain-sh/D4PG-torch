import argparse
import pathlib
import torch
import ray
import d4rl
import gym

from ray.util import ActorPool
from d4rl.utils.dataset_utils import TrajectoryDatasetWriter

from dataset_infos import CHECKPOINTS
from d4pg_torch.utils import read_config
from d4pg_torch.engine import Engine as D4PGEngine


@ray.remote
def collect(policy, n_episodes, env_id):
    total_timesteps = 0
    states, rewards, actions, dones = [], [], [], []
    env = gym.make(env_id)
    # observation, reward, done, action
    with torch.no_grad():
        for episode_th in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            while not done:
                normalized_obs = env.normalise_state(obs)
                action = policy.get_action(normalized_obs).cpu().numpy()
                next_obs, reward, done = env.step(action)
                episode_reward += reward
                # writer.append_data(s=obs, a=action, r=reward, done=done)
                states.append(obs)
                rewards.append(reward)
                actions.append(action)
                dones.append(done)
                obs = next_obs
                total_timesteps += 1
            # print(
            #     f"* [episode {episode_th + 1}/{n_episodes}] total reward: {episode_reward} total timesteps: {total_timesteps}"
            # )
    return states, rewards, actions, dones


D4RL_DATASET = pathlib.Path.home() / ".d4rl/datasets"

parser = argparse.ArgumentParser(description="Run training")
parser.add_argument("--config", type=str, help="Path to the config file.")
parser.add_argument("--collect_data", action="store_true")
parser.add_argument("--n_episodes", type=int, default=int(27e3))
parser.add_argument("--n_workers", type=int, default=10)
parser.add_argument("--seg", type=int, default=10)
parser.add_argument("--debug", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()
    config = read_config(args.config)
    if not args.collect_data:
        engine = D4PGEngine(config)
        engine.train()
    else:
        # collect data, load from checkpoint
        env_name = config["env"]
        checkpoint_path = CHECKPOINTS[f"DMC/{env_name}"]
        print(f"* load checkpoint from: {checkpoint_path}")
        source_device = config["device"]
        policy = torch.load(checkpoint_path, map_location=dict(source_device="cpu"))
        policy.to("cpu")

        writer = TrajectoryDatasetWriter(n_agent=1)
        if not ray.is_initialized():
            ray.init(local_mode=args.debug)
        actorpool = ActorPool([collect.remote for _ in range(args.n_workers)])
        tasks = [args.seg] * (args.n_episodes // args.seg)
        tails = args.n_episodes % args.seg
        if tails > 0:
            tasks.append(tails)
        results = actorpool.map(lambda actor, task: actor(policy, task, config), tasks)
        consum_episodes = 0
        for res in results:
            # append to writer
            for s, r, a, done in zip(*res):
                writer.append_data(s=s, a=a, r=r, done=done)
            consum_episodes += args.seg
            print(
                f"process: {consum_episodes}/{args.n_episodes} token num: {writer.num_tokens}"
            )
        fname = str(D4RL_DATASET / f"dmc-{env_name}.hdf5")
        writer.write_dataset(fname=fname)
        print(f"* data has been written to {fname}")
