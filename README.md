# gym-qubit

Qubit is an gym environment created for testing different algorithms for quantum control task.

## Installation

```bash
cd gym-qubit
pip install -e .
```

## Example of use

```python
>>> import gym
>>> import gym_qubit
>>> env = gym.make('Qubit-v0')
>>> emv.reset()
>>> env.reset()
[-0.003762385415214347, 1.3797619773722629e-07, 7.08729977089096e-14, 3.754345071879785e-07, 2.000000000000001, 0, -0.8097377327601498, 0.5867919598498706, 0.0, 0.0]
>>> env.step([0., 0.])
([3.0084781585190933e-07, 2.759048017167147e-07, 1.417459954178167e-13, -6.89892662841935e-07, 2.0000000000000013, 0.0, -0.31135039171109574, 0.9502951823414382, 0.0, 0.0], 0.0, False, {'fidelity': 0.0})
```
