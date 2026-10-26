# Double DQN (Dueling + Prioritised Replay) on Atari Pong

From-scratch PyTorch implementation, trained on `ALE/Pong-v5` for 5M environment steps (XJTLU DTS307TC coursework 2, individual work, June 2026).

- `network.py` Nature-DQN CNN with dueling value/advantage heads
- `agent.py` Double DQN target (online net selects, target net evaluates), Huber loss
- `replay_buffer.py` proportional prioritised replay (alpha 0.6, beta 0.4 -> 1.0)
- `env_wrappers.py` frame skip / max-pool, 84x84 grayscale, 4-frame stack, reward clipping, episodic life
- `train.py`, `evaluate.py`, `benchmark_sb3.py` (pretrained SB3 DQN reference), `plot.py`, `test_framework.py`

## Results (both numbers reported on purpose)
- Training: mean reward over the last 100 episodes stabilised at about +10 (human reference 9.3 used in the report).
- Formal evaluation, 30 episodes: **5.13 +/- 5.68**. For reference, a random agent scores about -20.6 and a pretrained SB3 DQN scores 21.0 in the same evaluation.

## Run (not re-verified)
```bash
pip install torch gymnasium ale-py tensorboard numpy matplotlib
python train.py
python evaluate.py --checkpoint checkpoints_per_v1/best_model.pth --episodes 30
python evaluate.py --random
```
Checkpoints, logs and the coursework report are not included.
