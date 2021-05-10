from .context import molecule_env
import gym


def test_example():
    env = molecule_env.MoleculeEnvironment()
    assert issubclass(type(env), gym.Env)
