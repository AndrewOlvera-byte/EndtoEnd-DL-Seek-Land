from __future__ import annotations

import subprocess
import sys


def main():
    # Quick sanity checks: BC and RL tiny runs + tiny tune sweeps
    cmds = [
        [sys.executable, "-m", "src.train_lightning", "task=bc", "+data.name=minari_kitchen_bc", "+model.name=bc_policy", "+loss.name=mse", "optim.max_epochs=1", "io.batch_size=64"],
        [sys.executable, "-m", "src.train_lightning", "+exp=rl_ppo_cartpole", "optim.max_epochs=1"],
        [sys.executable, "-m", "src.tune_search", "bc", "--debug", "--topk", "1", "--run-topk"],
        [sys.executable, "-m", "src.tune_search", "rl", "--debug", "--topk", "1", "--run-topk"],
    ]
    for i, cmd in enumerate(cmds, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(cmds)}: {' '.join(cmd[2:])}")
        print('='*80)
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()


