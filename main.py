from ga.train import ga_runner
from env.env import Env
import yaml

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        args = yaml.safe_load(f)
    
    env = Env(args)
    
    if args['algo'] == 'ga':
        runner = ga_runner
    else:
        raise NotImplementedError()
    
    runner(args, env)