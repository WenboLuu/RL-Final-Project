import wandb


def initialize_wandb(project_name, config):
    """
    Initializes Weights & Biases with the given project name and configuration.
    """
    wandb.init(project=project_name, config=config)
    return wandb.config


def log_metrics(metrics, step=None):
    """
    Logs metrics to Weights & Biases.

    Parameters:
    - metrics: A dictionary of metric names and values.
    - step: The global step at which the metrics are logged.
    """
    wandb.log(metrics, step=step)


def finalize_wandb():
    """
    Finalizes the Weights & Biases run.
    """
    wandb.finish()
