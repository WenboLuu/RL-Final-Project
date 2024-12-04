import wandb
import yaml
from main import main

# Load the sweep configuration from the YAML file
with open("../sweep_config.yaml") as file:
    sweep_config = yaml.safe_load(file)

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, entity="fruitswordman", project="RL-Final-Project")

# Define the function to run for each sweep
def sweep_train():
    with wandb.init() as run:
        config = wandb.config

        # Get the seed from the config
        seed = config.seed

        main(seed)

# Run the sweep
wandb.agent(sweep_id, function=sweep_train)
