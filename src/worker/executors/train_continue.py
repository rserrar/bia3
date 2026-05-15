from .train import execute_train


def execute_train_continue(payload: dict) -> dict:
    return execute_train(payload)
