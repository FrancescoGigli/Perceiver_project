# src/utils/scheduler.py
# Learning rate scheduler utilities.

import torch
import torch.optim as optim

def get_scheduler(optimizer, scheduler_name="cosine", total_epochs=100, lr_step_size=30, lr_gamma=0.1, eta_min=0):
    """
    Returns a learning rate scheduler.
    Args:
        optimizer: The optimizer.
        scheduler_name (str): Name of the scheduler ('cosine', 'step', or 'multistep').
        total_epochs (int): Total number of epochs for CosineAnnealingLR.
        lr_step_size (int): Step size for StepLR.
        lr_gamma (float): Gamma factor for StepLR and MultiStepLR.
        eta_min (float): Minimum learning rate for CosineAnnealingLR.
    Returns:
        A PyTorch learning rate scheduler.
    """
    scheduler_name = scheduler_name.lower()

    if scheduler_name == "cosine":
        print(f"Using CosineAnnealingLR scheduler with T_max={total_epochs}, eta_min={eta_min}.")
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min)

    elif scheduler_name == "step":
        print(f"Using StepLR scheduler with step_size={lr_step_size}, gamma={lr_gamma}.")
        return optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    elif scheduler_name == "multistep":
        milestones = [84, 102, 114]
        print(f"Using MultiStepLR scheduler with milestones={milestones}, gamma={lr_gamma}.")
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=lr_gamma)

    else:
        print(f"Scheduler '{scheduler_name}' not recognized. Using no scheduler.")
        return None  # Or fallback: optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

if __name__ == '__main__':
    print("Scheduler module test: OK")
