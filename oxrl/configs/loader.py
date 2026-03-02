"""
Config loading: YAML -> validated Config object.

Single entry point: load_and_verify().
"""
import yaml
import sys
from pydantic import ValidationError

from oxrl.configs.schema import Config
from oxrl.configs.sync import sync_deepspeed_config


def load_and_verify(
    method: str,
    input_yaml: str,
    experiment_id: str,
    world_size: int | None = None,
) -> Config:
    '''
        method: "sl" or "rl"
        input_yaml: path to the yaml file
        experiment_id: experiment identifier
        world_size: number of GPUs for SL training (optional for RL)
    '''
    try:
        with open(input_yaml, "r") as f:
            raw_config = yaml.safe_load(f)

        # now verify the config
        config = Config(**raw_config)
        config.run.method = method
        # Update Run details
        config.run.experiment_id = experiment_id

        # Determine world_size based on method
        if method == "sl":
            if world_size is None:
                raise ValueError("world_size must be specified for SL training")

        elif method == "rl":
            world_size = config.run.training_gpus

        # Sync AFTER updating world_size
        sync_deepspeed_config(config, world_size)

        print( "\n" + 20*"=" + "Config" + 20*"=")
        print(f"Contents of {input_yaml}")
        print(config.model_dump_json(indent=4))
        print(46*"=")

    except ValidationError as e:
        print("Configuration Error:")
        print(e)
        sys.exit(1)

    except FileNotFoundError:
        print("Error: Config file not found.")
        sys.exit(1)

    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)

    return config
