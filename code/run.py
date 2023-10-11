from ga.train import ga_runner
from rl.train import rl_runner
from env.env import Env
import yaml

if __name__ == '__main__':
    with open('conf.yaml', 'r') as f:
        args = yaml.safe_load(f)
    
    env = Env(args)

    if args['algo'] == 'ga':
        runner = ga_runner
    else:
        runner = rl_runner
    runner(args, env)