# ViTCIFAR-Lightning-Example
Repo displaying good pipeline design using lightning to focus on implementation rather than boilerplate

## Run in Docker

Build the image (first time or after dependency changes):

```bash
docker compose build --no-cache
```

Start a training run with the base defaults:

```bash
docker compose run --rm train-lightning
```

### Switch experiments (Hydra groups)

Experiments are Hydra groups registered via ConfigStore. Select them by name:

```bash
# Default (vit_cifar10)
docker compose run --rm train-lightning

# Quick debug run (smaller batch, few epochs)
docker compose run --rm train-lightning exp=quick_debug
```

You can also override any field at the CLI:

```bash
docker compose run --rm train-lightning exp=vit_cifar10 io.batch_size=256 trainer.devices=1
```

### Registry-based components

Models, datamodules, optimizers, schedulers, and losses are resolved via simple registries with decorators:

```python
from src.registry import register_model

@register_model("vit")
class VisionTransformer(nn.Module):
    ...
```

- Model: `model.name=vit`
- Data: `data.name=cifar10`
- Optimizer: `optim.name=adamw`
- Scheduler: `sched.name=cosine`
- Loss: `loss.name=cross_entropy`

This avoids fragile `_target_` strings and makes swapping components trivial.

### Notes

- Hydra output directories default under `./output` or `${OUTPUT_DIR}` if set.
- The script prints the resolved config on start.
- For a full error stack, set `HYDRA_FULL_ERROR=1`.
