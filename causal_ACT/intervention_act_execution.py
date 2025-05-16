from pprint import pprint
import os
import pickle
import numpy as np
import argparse
from time import perf_counter

import torch
from sklearn.linear_model import Ridge
from torch.distributions import Bernoulli

from policy import ACTPolicy
from sim_env import make_sim_env
from utils import set_seed
from utils import sample_box_pose
from constants import SIM_TASK_CONFIGS
from sim_env import BOX_POSE
from einops import rearrange



import time
TIME_START = time.time()

N_ROLLOUTS_PER_ITERATE = 1
print(f"{N_ROLLOUTS_PER_ITERATE=}")


def sample(weights, temperature):
    return Bernoulli(logits=torch.from_numpy(weights) / temperature).sample().long().numpy()


def linear_regression(masks, rewards, alpha=1.0):
    # breakpoint()
    stacked_masks = np.stack([arr.reshape(-1) for arr in masks])
    model = Ridge(alpha).fit(stacked_masks, rewards)
    return model.coef_, model.intercept_


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image



class SoftQAlgo:
    def __init__(
            self,
            num_dims,
            reward_fn,
            its,
            temperature=1.0,
            device=None,
            evals_per_it=1,
    ):
        self.num_dims = num_dims
        self.reward_fn = reward_fn
        self.its = its
        self.device = device
        self.temperature = lambda t: temperature
        self.evals_per_it = evals_per_it

    def run(self):
        t = self.temperature(0)
        weights = np.zeros(self.num_dims)

        trace = []
        masks = []
        rewards = []
        for it in range(self.its):
            start = perf_counter()
            mask = sample(weights, t)
            reward = np.mean([self.reward_fn(mask,N_ROLLOUTS_PER_ITERATE) for _ in range(self.evals_per_it)])
            masks.append(mask)
            rewards.append(reward)
            
            weights, _ = linear_regression(masks, rewards, alpha=1.0)
            weights = weights.reshape(self.num_dims) # reshape back to the same dim with src

            trace.append(
                {
                    "it": it,
                    "reward": reward,
                    "mask": mask,
                    "weights": weights,
                    "mode": (np.sign(weights).astype(np.int64) + 1) // 2,
                    "time": perf_counter() - start,
                    "past_mean_reward": np.mean(rewards),
                }
            )
            # pprint(trace[-1])

        return trace

def prepare_configs(args):

    set_seed(args['seed'])
    # command line parameters
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    task_name = args['task_name']

    # get task parameters
    task_config = SIM_TASK_CONFIGS[task_name]

    dataset_dir = task_config['dataset_dir']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    print(f"-----------Loading data from {dataset_dir}---------")

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    else:
        raise NotImplementedError

    config = {
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        #config for intervention
        'num_its': args['num_its'],
        'temperature': args['temperature']
    }

    return config


def intervention_policy_execution(config):
    # parameters:
    set_seed(config['seed'])
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    
    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    env = make_sim_env(task_name)
    env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']
    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    # def run_step(mask):
    #     trajectories = run_fixed_mask(env, policy_model, state_encoder, mask, 1)
    #     return Trajectory.reward_sum_mean(trajectories)
    def mean_episodic_sum_rewards(mask, num_episodes):

        returns_per_episode = []
        for rollout_id in range(num_episodes):
            rewards_per_step = []
            BOX_POSE[0] = sample_box_pose()
            ts = env.reset()
            ### evaluation loop
            if temporal_agg:
                all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

            with torch.inference_mode():
                for t in range(max_timesteps):

                    obs = ts.observation
                    qpos_numpy = np.array(obs['qpos'])
                    qpos = pre_process(qpos_numpy)
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    curr_image = get_image(ts, camera_names)

                    ### query policy
                    if config['policy_class'] == "ACT":
                        if t % query_frequency == 0:
                            all_actions = policy(qpos, curr_image, mask)
                        if temporal_agg:
                            all_time_actions[[t], t:t+num_queries] = all_actions
                            actions_for_curr_step = all_time_actions[:, t]
                            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
                            k = 0.01
                            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                            exp_weights = exp_weights / exp_weights.sum()
                            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                        else:
                            raw_action = all_actions[:, t % query_frequency]
                    else:
                        raise NotImplementedError

                    ### post-process actions
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = post_process(raw_action)
                    target_qpos = action

                    ### step the environment
                    ts = env.step(target_qpos)

                    ### for visualization
                    rewards_per_step.append(ts.reward)
            # after one episode:
            # if rewards_per_step[:-1] == env_max_reward:
            #     rewards_per_step.append(100)
            rewards_per_step = np.array(rewards_per_step)
            returns_per_episode.append(np.sum(rewards_per_step[rewards_per_step!=None]))
            print(f"rollout {rollout_id}: sum of episodic rewards: {np.sum(rewards_per_step[rewards_per_step!=None])}")
        return np.mean(returns_per_episode).item()

    # mask size: [512,15,20]
    trace = SoftQAlgo((512,15,20), mean_episodic_sum_rewards, config['num_its'], temperature=config['temperature']).run()

    best_mask = trace[-1]['mode']
    best_returns = mean_episodic_sum_rewards(best_mask, 20)
    # save best mask:
    N_disturbs = (len(env.physics.data.qpos)-23)//7
    N_ones_in_mask = best_mask.sum()
    # breakpoint()
    np.save(f"{ckpt_dir}/best_mask_{config['num_its']}iterates_{N_ROLLOUTS_PER_ITERATE}rollperiters_w_{N_disturbs}ndisturbs_{N_ones_in_mask}ones_seed{config['seed']}_temp{config['temperature']}_{((time.time()-TIME_START)/3600.0):.2f}hours.npy", best_mask)
    print(f"Final mask {best_mask.tolist()}")
    print(f"Final reward {best_returns}")
    print(f"Intervention {config['num_its']} iterates in sim with {N_disturbs} disturbs")
    print(f"{N_ROLLOUTS_PER_ITERATE=}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_its', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=10)

    # arguments for ACT
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    # main(vars(parser.parse_args()))
    configs = prepare_configs(vars(parser.parse_args()))
    intervention_policy_execution(configs)

    #  Training Summary
    local_start = time.localtime(TIME_START)
    local_end = time.localtime(time.time())
    trainint_time = (time.time()-TIME_START)/3600.0

    print(f"start at: {local_start.tm_mday}/{local_start.tm_mon}/{local_start.tm_year}: \
                {local_start.tm_hour}:{local_start.tm_min}:{local_start.tm_sec}")
    print(f"end   at: {local_end.tm_mday}/{local_end.tm_mon}/{local_end.tm_year}: \
        {local_end.tm_hour}:{local_end.tm_min}:{local_end.tm_sec}")
    print(f"in total: {trainint_time} hours")


if __name__ == '__main__':
    main()
