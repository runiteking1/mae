"""MLflow logging utilities. Always active on rank 0."""

import util.misc as misc


def _make_run_name(args):
    """Build run name from ablation variables: model, optimizer, lr."""
    model = getattr(args, 'model', 'unknown')
    optimizer = getattr(args, 'optimizer', 'adamw')
    lr = getattr(args, 'lr', None) or getattr(args, 'blr', None) or 0
    return f"{model}_opt-{optimizer}_lr-{lr:.1e}"


def init_mlflow(args, experiment_name="mae"):
    """Initialize MLflow run on rank 0. Returns True if active."""
    if not misc.is_main_process():
        return False

    import mlflow
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=_make_run_name(args))
    params = vars(args)
    for k, v in params.items():
        mlflow.log_param(k, str(v)[:500])
    return True


def log_metrics(metrics: dict, step: int, active: bool):
    """Log a dict of metrics at the given step."""
    if not active:
        return
    import mlflow
    mlflow.log_metrics(metrics, step=step)


def end_mlflow(active: bool):
    """End the MLflow run."""
    if not active:
        return
    import mlflow
    mlflow.end_run()
