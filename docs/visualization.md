## TensorBoard

TensorBoard logs are written under:

```text
results/{run_name}/tb/
```

Launch TensorBoard:

```bash
cd ~/repos/3rgs
source .venv/bin/activate
tensorboard --logdir results/tennis_court_smoke/tb --port 6006
```

Open:

```text
http://localhost:6006/
```