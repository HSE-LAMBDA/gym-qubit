# gym-qubit

Qubit is an gym environment created for testing different algorithms for quantum control task.

## Installation

```bash
cd gym-qubit
pip install -e .
```

## Example of use

```python
>> from gym_qubit import envs
>> qubit = envs.TransmonEnv()
>> qubit.step([0, 0])
([3.0084781585190933e-07, 2.759048017167147e-07, 1.417459954178167e-13, -6.89892662841935e-07, 2.0000000000000013, 0.0, -0.31135039171109574, 0.9502951823414382, 0, 0], 0.0, False, {'fidelity': 0.0})
```
