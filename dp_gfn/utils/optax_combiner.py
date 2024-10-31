import optax

def combine_optax_transforms(
    learning_rate=1e-3,
    clip_grad_norm=1.0,
    optimizer_name="adam",
    weight_decay=0.0,
    lr_scheduler_type="constant",
    lr_scheduler_kwargs={},
    multi_transform_dict=None,
):
    """Combines multiple optax gradient transformations.

    Args:
        learning_rate (float): The learning rate for the optimizer.
        clip_grad_norm (float): The maximum norm for gradient clipping.
        optimizer_name (str, optional): The name of the optimizer to use. 
                                        Defaults to "adam".
        weight_decay (float, optional): Weight decay coefficient. Defaults to 0.0.
        lr_scheduler_type (str, optional): The type of learning rate scheduler to use. 
                                            Defaults to "constant".
        lr_scheduler_kwargs (dict, optional): Keyword arguments for the learning rate scheduler. 
                                                Defaults to {}.
        multi_transform_dict (dict, optional): A dictionary specifying separate optimizers 
                                                for different parameter groups. Defaults to None.

    Returns:
        optax.GradientTransformation: The combined optax gradient transformation.

    Example:
        ```python
        # Example usage with Adam optimizer, gradient clipping, and a linear scheduler
        tx = combine_optax_transforms(
            learning_rate=1e-3,
            clip_grad_norm=1.0,
            optimizer_name="adam",
            lr_scheduler_type="linear",
            lr_scheduler_kwargs={"init_value": 1e-3, "end_value": 1e-5, "transition_steps": 1000},
        )

        # Example usage with separate optimizers for different parameter groups
        tx = combine_optax_transforms(
            learning_rate=1e-3,
            clip_grad_norm=1.0,
            multi_transform_dict={
                "encoder": {"optimizer_name": "adam", "weight_decay": 0.1},
                "decoder": {"optimizer_name": "sgd", "learning_rate": 1e-2},
            },
        )
        ```
    """

    # Choose optimizer
    if optimizer_name.lower() == "adam":
        optimizer = optax.adam(learning_rate)
    elif optimizer_name.lower() == "sgd":
        optimizer = optax.sgd(learning_rate, momentum=0.9)
    elif optimizer_name.lower() == "adabelief":
        optimizer = optax.adabelief(learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Add weight decay if specified
    if weight_decay > 0:
        optimizer = optax.add_decayed_weights(optimizer, weight_decay)

    # Create learning rate scheduler
    if lr_scheduler_type.lower() == "constant":
        lr_scheduler = optax.constant_schedule(learning_rate)
    elif lr_scheduler_type.lower() == "linear":
        lr_scheduler = optax.linear_schedule(**lr_scheduler_kwargs)
    elif lr_scheduler_type.lower() == "cosine":
        lr_scheduler = optax.cosine_decay_schedule(**lr_scheduler_kwargs)
    elif lr_scheduler_type.lower() == "warmup_cosine":
        lr_scheduler = optax.warmup_cosine_decay_schedule(**lr_scheduler_kwargs)
    else:
        raise ValueError(f"Unsupported learning rate scheduler: {lr_scheduler_type}")

    # Combine transformations
    if multi_transform_dict is not None:
        # Create separate optimizers for different parameter groups
        transforms = {}
        for group_name, group_params in multi_transform_dict.items():
            group_optimizer = combine_optax_transforms(
                learning_rate=group_params.get("learning_rate", learning_rate),
                clip_grad_norm=clip_grad_norm,
                optimizer_name=group_params.get("optimizer_name", optimizer_name),
                weight_decay=group_params.get("weight_decay", weight_decay),
                lr_scheduler_type=lr_scheduler_type,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
            )
            transforms[group_name] = group_optimizer
        tx = optax.multi_transform(transforms, group_names=list(transforms.keys()))
    else:
        # Combine optimizer, scheduler, and gradient clipping
        tx = optax.chain(
            optax.clip_by_global_norm(clip_grad_norm),
            optax.scale_by_schedule(lr_scheduler),
            optimizer,
        )

    return tx

tx = combine_optax_transforms(
        lr_scheduler_type="linear",
        lr_scheduler_kwargs={"init_value": 1e-3, "end_value": 1e-5, "transition_steps": 1000},
        multi_transform_dict={            
            "encoder": {"optimizer_name": "adam", "weight_decay": 0.1},
            "decoder": {"optimizer_name": "sgd", "learning_rate": 1e-2},
        },
    )