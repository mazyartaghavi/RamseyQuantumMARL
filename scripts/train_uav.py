import argparse, yaml, os, time
from qio_marl.algos.qio_marl import QIOMARL
from qio_marl.envs.forest_uav import ForestUAV
from qio_marl.utils.logger import CSVLogger

def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join("runs", timestamp)
    os.makedirs(out_dir, exist_ok=True)
    logger = CSVLogger(os.path.join(out_dir, "train_log.csv"))

    env = ForestUAV(
        grid_size=cfg["env"]["grid_size"],
        n_agents=cfg["env"]["n_agents"],
        obs_radius=cfg["env"]["obs_radius"],
        episode_len=cfg["env"]["episode_len"],
        seed=cfg["env"]["seed"],
        comm_budget=cfg["env"]["comm_budget"],
    )

    algo = QIOMARL(cfg, env.observation_dim, env.action_dim, n_agents=env.n_agents, out_dir=out_dir)
    total_steps = cfg["train"]["total_steps"]
    log_interval = cfg["train"]["log_interval"]
    eval_interval = cfg["train"]["eval_interval"]

    obs, _ = env.reset()
    ep_reward = 0.0
    ep_len = 0
    episode = 0

    for step in range(1, total_steps + 1):
        actions, logp, value, extra = algo.act(obs)
        next_obs, reward, done, info = env.step(actions)
        algo.store(obs, actions, reward, done, logp, value, extra)
        obs = next_obs
        ep_reward += reward
        ep_len += 1

        if algo.ready_to_update():
            metrics = algo.update()
            metrics.update({"global_step": step})
            logger.log(metrics)

        if done:
            logger.log({"episode": episode, "ep_return": ep_reward, "ep_len": ep_len, "global_step": step})
            obs, _ = env.reset()
            ep_reward, ep_len = 0.0, 0
            episode += 1

        if step % log_interval == 0:
            print(f"[{step}] recent return={ep_reward:.2f} H={algo.last_entropy:.3f} alpha={algo.last_alpha:.4f}")

        if step % eval_interval == 0:
            eval_ret = 0.0
            for _ in range(cfg["train"]["eval_episodes"]):
                o, _ = env.reset(eval_mode=True)
                done_eval = False
                while not done_eval:
                    a, *_ = algo.act(o, greedy=True)
                    o, r, done_eval, _ = env.step(a)
                    eval_ret += r
            eval_ret /= cfg["train"]["eval_episodes"]
            logger.log({"global_step": step, "eval_return": eval_ret})

    logger.close()
    print(f"Training complete. Logs at: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/uav_default.yaml")
    args = parser.parse_args()
    main(args.config)
