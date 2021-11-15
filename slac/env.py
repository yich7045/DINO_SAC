import gym
import minitouch.env

def make_dmc():
    env = gym.make("PushingDebug-v0")
    return env
