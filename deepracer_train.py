import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import deepracer_gym
from collections import deque
import random
import os
import pickle
import json

# =====================================
# DEVICE SETUP
# =====================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================
# CHECKPOINT DIRECTORY
# =====================================
CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)


# =====================================
# Preprocess observation (CPU ONLY)
# =====================================
def preprocess_obs(obs):
    cam = obs["STEREO_CAMERAS"].astype(np.float32) / 255.0      # (2,120,160)
    lidar = np.nan_to_num(obs["LIDAR"].astype(np.float32), posinf=10.0)
    return cam, lidar


# =====================================
# Actor Network
# =====================================
class Actor(nn.Module):
    def __init__(self, action_dim=2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 2, 120, 160)
            cnn_out_dim = self.cnn(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)

    def forward(self, cam, lidar):
        x = self.cnn(cam)
        x = torch.cat([x, lidar], dim=1)
        x = self.fc(x)
        return self.mean(x), self.log_std(x).clamp(-20, 2)


# =====================================
# Critic Network
# =====================================
class Critic(nn.Module):
    def __init__(self, action_dim=2):
        super().__init__()

        self.cnn = Actor().cnn

        with torch.no_grad():
            dummy = torch.zeros(1, 2, 120, 160)
            vision_out = self.cnn(dummy).shape[1]

        input_dim = vision_out + 64 + action_dim

        self.q = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, cam, lidar, action):
        v = self.cnn(cam)
        x = torch.cat([v, lidar, action], dim=1)
        return self.q(x)


# =====================================
# Replay Buffer (Stores CPU numpy)
# =====================================
class ReplayBuffer:
    def __init__(self, size=200000):
        self.buffer = deque(maxlen=size)

    def add(self, cam, lidar, action, reward, next_cam, next_lidar, done):
        self.buffer.append((
            (cam * 255).astype(np.uint8),
            lidar.astype(np.float32),
            np.array(action, np.float32),
            float(reward),
            (next_cam * 255).astype(np.uint8),
            next_lidar.astype(np.float32),
            bool(done),
        ))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)

        cams, lidars, acts, rews, ncams, nlids, dones = zip(*batch)

        # ---- FAST batch stacking ----
        cams  = torch.from_numpy(np.stack(cams, axis=0)).to(device).float() / 255.0
        ncams = torch.from_numpy(np.stack(ncams, axis=0)).to(device).float() / 255.0

        lidars = torch.from_numpy(np.stack(lidars, axis=0)).to(device).float()
        nlids  = torch.from_numpy(np.stack(nlids,  axis=0)).to(device).float()

        acts = torch.from_numpy(np.stack(acts, axis=0)).to(device).float()
        rews = torch.tensor(rews, device=device, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, device=device, dtype=torch.float32).unsqueeze(1)

        return cams, lidars, acts, rews, ncams, nlids, dones


    def __len__(self):
        return len(self.buffer)


# =====================================
# SAC
# =====================================
class SAC:
    def __init__(self, action_dim=2, lr=3e-4, gamma=0.99, tau=0.005):
        self.actor = Actor(action_dim).to(device)
        self.critic1 = Critic(action_dim).to(device)
        self.critic2 = Critic(action_dim).to(device)
        self.target1 = Critic(action_dim).to(device)
        self.target2 = Critic(action_dim).to(device)

        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        self.actor_opt   = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau

        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -action_dim

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ------------------------
    # ACTION SAMPLING
    # ------------------------
    def sample_action(self, cam_np, lidar_np):
        cam = torch.tensor(cam_np, device=device).unsqueeze(0)
        lidar = torch.tensor(lidar_np, device=device).unsqueeze(0)

        mean, log_std = self.actor(cam, lidar)
        std = log_std.exp()

        noise = torch.randn_like(mean)
        action = torch.tanh(mean + std * noise)
        return action.detach().cpu().numpy()[0]

    # ------------------------
    # UPDATE
    # ------------------------
    def update(self, replay, batch_size=64):
        if len(replay) < batch_size:
            return None

        cam, lidar, act, rew, next_cam, next_lidar, done = replay.sample(batch_size)

        # ---------- Critic ----------
        with torch.no_grad():
            nm, nls = self.actor(next_cam, next_lidar)
            ns = nls.exp()
            noise = torch.randn_like(nm)
            na = torch.tanh(nm + ns * noise)

            logp = (-0.5*(noise**2 + 2*nls + np.log(2*np.pi))).sum(1, keepdim=True)
            logp -= torch.log(1 - na.pow(2) + 1e-7).sum(1, keepdim=True)

            tq1 = self.target1(next_cam, next_lidar, na)
            tq2 = self.target2(next_cam, next_lidar, na)
            tq  = torch.min(tq1, tq2) - self.alpha * logp

            target = rew + (1 - done) * self.gamma * tq

        q1 = self.critic1(cam, lidar, act)
        q2 = self.critic2(cam, lidar, act)

        critic1_loss = F.mse_loss(q1, target)
        critic2_loss = F.mse_loss(q2, target)

        self.critic1_opt.zero_grad(); critic1_loss.backward(); self.critic1_opt.step()
        self.critic2_opt.zero_grad(); critic2_loss.backward(); self.critic2_opt.step()

        # ---------- Actor ----------
        mean, log_std = self.actor(cam, lidar)
        std = log_std.exp()
        noise = torch.randn_like(mean)
        action = torch.tanh(mean + std * noise)

        logp = (-0.5*(noise**2 + 2*log_std + np.log(2*np.pi))).sum(1, keepdim=True)
        logp -= torch.log(1 - action.pow(2) + 1e-7).sum(1, keepdim=True)

        q1_pi = self.critic1(cam, lidar, action)
        q2_pi = self.critic2(cam, lidar, action)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * logp - q_pi).mean()

        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        # ---------- Entropy α ----------
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        # ---------- Target update ----------
        self._soft_update(self.critic1, self.target1)
        self._soft_update(self.critic2, self.target2)

        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "alpha_loss": alpha_loss.item()
        }

    def _soft_update(self, src, tgt):
        for p, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)


# =====================================
# SAVE / LOAD CHECKPOINTS
# =====================================
def save_checkpoint(agent, replay, global_step, episode):
    path = f"{CKPT_DIR}/checkpoint.pt"

    ckpt = {
        "actor": agent.actor.state_dict(),
        "critic1": agent.critic1.state_dict(),
        "critic2": agent.critic2.state_dict(),
        "target1": agent.target1.state_dict(),
        "target2": agent.target2.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "actor_opt": agent.actor_opt.state_dict(),
        "critic1_opt": agent.critic1_opt.state_dict(),
        "critic2_opt": agent.critic2_opt.state_dict(),
        "alpha_opt": agent.alpha_opt.state_dict(),
        "global_step": global_step,
        "episode": episode,
    }

    torch.save(ckpt, path)

    # Save replay buffer separately
    with open(f"{CKPT_DIR}/replay.pkl", "wb") as f:
        pickle.dump(replay.buffer, f)

    print(f"[Checkpoint Saved] Episode={episode} Step={global_step}")


def load_checkpoint(agent, replay):
    path = f"{CKPT_DIR}/checkpoint.pt"
    if not os.path.exists(path):
        print("No checkpoint found — training from scratch.")
        return 0, 0

    ckpt = torch.load(path, map_location=device)

    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic1.load_state_dict(ckpt["critic1"])
    agent.critic2.load_state_dict(ckpt["critic2"])
    agent.target1.load_state_dict(ckpt["target1"])
    agent.target2.load_state_dict(ckpt["target2"])

    agent.log_alpha.data = ckpt["log_alpha"].to(device)

    agent.actor_opt.load_state_dict(ckpt["actor_opt"])
    agent.critic1_opt.load_state_dict(ckpt["critic1_opt"])
    agent.critic2_opt.load_state_dict(ckpt["critic2_opt"])
    agent.alpha_opt.load_state_dict(ckpt["alpha_opt"])

    # Load replay buffer
    replay_path = f"{CKPT_DIR}/replay.pkl"
    if os.path.exists(replay_path):
        with open(replay_path, "rb") as f:
            replay.buffer = pickle.load(f)

    print(f"[Checkpoint Loaded] Episode={ckpt['episode']} Step={ckpt['global_step']}")

    return ckpt["global_step"], ckpt["episode"]


# =====================================
# PARALLEL ENV
# =====================================
class ParallelDeepRacerEnv:
    def __init__(self, ports):
        self.envs = [gym.make("deepracer-v0", host="127.0.0.1", port=p) for p in ports]
        self.num_envs = len(self.envs)

    def reset(self):
        return [env.reset()[0] for env in self.envs]

    def step(self, actions):
        obs, rews, dones, truncs = [], [], [], []
        for env, act in zip(self.envs, actions):
            o, r, d, t, _ = env.step(act)
            obs.append(o); rews.append(r); dones.append(d); truncs.append(t)
        return obs, rews, dones, truncs

    def reset_single(self, i):
        return self.envs[i].reset()[0]


# =====================================
# TRAIN LOOP
# =====================================

# ports = [8891, 8892]   # Expand if needed
ports = [8891, 8892, 8893, 8894, 8895, 8896, 8897, 8898]
envs = ParallelDeepRacerEnv(ports)

agent = SAC()
replay = ReplayBuffer()
writer = SummaryWriter("runs/deepracer_sac_parallel_with_saving")

# Load checkpoint if exists
global_step, start_episode = load_checkpoint(agent, replay)

episodes = 2000
max_steps = 500

obs_list = envs.reset()
cam_list = []
lid_list = []

for obs in obs_list:
    c, l = preprocess_obs(obs)
    cam_list.append(c)
    lid_list.append(l)

# =====================================
# MAIN LOOP
# =====================================
for ep in range(start_episode, episodes):
    ep_rewards = [0.0] * envs.num_envs

    for step in range(max_steps):

        # 1. Select actions
        actions = []
        for c, l in zip(cam_list, lid_list):
            a = agent.sample_action(c, l)
            actions.append([float(a[0]), float((a[1] + 1) / 2)])

        # 2. Env step
        next_obs_list, rewards, dones, truncs = envs.step(actions)

        next_cam_list = []
        next_lid_list = []

        for i in range(envs.num_envs):
            next_cam, next_lid = preprocess_obs(next_obs_list[i])
            next_cam_list.append(next_cam)
            next_lid_list.append(next_lid)

            replay.add(cam_list[i], lid_list[i], actions[i], rewards[i],
                       next_cam, next_lid, dones[i] or truncs[i])

            ep_rewards[i] += rewards[i]

        cam_list = next_cam_list
        lid_list = next_lid_list

        # 3. SAC update
        losses = agent.update(replay)
        if losses:
            for k, v in losses.items():
                writer.add_scalar(f"loss/{k}", v, global_step)

        # 4. Reset done envs
        for i in range(envs.num_envs):
            if dones[i] or truncs[i]:
                o = envs.reset_single(i)
                cam_list[i], lid_list[i] = preprocess_obs(o)

        global_step += 1

        # 🔥 Auto-save every 5000 steps
        if global_step % 5000 == 0:
            save_checkpoint(agent, replay, global_step, ep)

    mean_reward = float(np.mean(ep_rewards))
    writer.add_scalar("episode/mean_reward", mean_reward, ep)
    print(f"Episode {ep} | Mean Reward: {mean_reward:.2f}")

    # 🔥 Save at end of each episode
    save_checkpoint(agent, replay, global_step, ep)

writer.close()
