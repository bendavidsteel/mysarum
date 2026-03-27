#!/usr/bin/env python3
"""
GRA Parameter Search — Hydra entry point.

Usage:
    # Default search
    python run.py

    # Override settings
    python run.py search.batch_size=64 search.num_generations=200

    # Quick test
    python run.py search.num_generations=10 search.batch_size=32 search.sim_steps=100

    # Disable wandb
    python run.py wandb.enabled=false

    # Replay saved archive
    python run.py mode=replay replay_path=archive.pkl
"""

import pickle
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Print resolved config
    print(OmegaConf.to_yaml(cfg, resolve=True))

    mode = cfg.get("mode", "search")

    if mode == "replay":
        replay(cfg)
    else:
        search(cfg)


def search(cfg: DictConfig):
    sim_type = cfg.sim.get("type", "continuous")

    if sim_type == "discrete":
        from search_discrete import run_search, plot_archive, show_top_results
        import sim_discrete as sim_mod
        sim_mod.configure(cfg)
        archive = run_search(cfg)

        save_path = Path("archive.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(archive, f)
        print(f"\nArchive saved to {save_path}")

        plot_archive(archive, save_path="archive.png")
        show_top_results(archive, n=cfg.viz.top_n,
                         graph=cfg.sim.get("graph", "petersen"),
                         sim_steps=cfg.viz.replay_steps,
                         save_dir=cfg.viz.save_dir)
    else:
        from search import run_search
        archive = run_search(cfg)

        save_path = Path("archive.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(archive, f)
        print(f"\nArchive saved to {save_path}")

        from viz import plot_archive, show_top_results
        plot_archive(archive, save_path="archive.png")

        import sim as sim_mod
        sim_mod.configure(cfg)
        show_top_results(archive, n=cfg.viz.top_n,
                         graph_idx=cfg.search.graph_idx,
                         sim_steps=cfg.viz.replay_steps,
                         save_dir=cfg.viz.save_dir)


def replay(cfg: DictConfig):
    replay_path = cfg.get("replay_path", "archive.pkl")
    print(f"Loading archive from {replay_path}...")
    with open(replay_path, "rb") as f:
        archive = pickle.load(f)

    sim_type = cfg.sim.get("type", "continuous")

    if sim_type == "discrete":
        import sim_discrete as sim_mod
        sim_mod.configure(cfg)
        from search_discrete import configure_search, plot_archive, show_top_results
        configure_search(cfg)
        plot_archive(archive, save_path="archive.png")
        show_top_results(archive, n=cfg.viz.top_n,
                         graph=cfg.sim.get("graph", "petersen"),
                         sim_steps=cfg.viz.replay_steps,
                         save_dir=cfg.viz.save_dir)
    else:
        import sim as sim_mod
        sim_mod.configure(cfg)
        from search import configure_search
        configure_search(cfg)
        from viz import plot_archive, show_top_results, animate_simulation
        import jax

        plot_archive(archive, save_path="archive.png")

        if cfg.viz.animate:
            top = archive.best_cells(cfg.viz.top_n)
            for rank, (r, c, fitness) in enumerate(top):
                params = jax.tree.map(lambda arr: arr[r, c], archive.params)
                print(f"\nAnimating rank {rank+1} (fitness={fitness:.2f})...")
                animate_simulation(params, graph_idx=cfg.search.graph_idx,
                                   num_steps=cfg.viz.replay_steps,
                                   save_path=f"anim_rank{rank+1}.gif")
        else:
            show_top_results(archive, n=cfg.viz.top_n,
                             graph_idx=cfg.search.graph_idx,
                             sim_steps=cfg.viz.replay_steps,
                             save_dir=cfg.viz.save_dir)


if __name__ == "__main__":
    main()
