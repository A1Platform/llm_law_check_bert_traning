import wandb

def is_wandb_logged_in():
    try:
        return wandb.api.api_key is not None
    except:
        return False


print(f"is_wandb_logged_in: {is_wandb_logged_in()}")
