import gym
from gym.envs.registration import register

register(
    id='DonStarveEnv-v0',
    entry_point='donstarve_env:DonStarveEnv',
)