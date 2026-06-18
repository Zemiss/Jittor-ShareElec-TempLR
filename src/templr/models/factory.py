def build_model(args, node_size):
    model_name = args.model_name.lower()
    if model_name == 'baseline':
        from .baseline import build_baseline_model
        return build_baseline_model(args, node_size)
    if model_name == 'mynet':
        from .mynet import build_mynet_model
        return build_mynet_model(args, node_size)

    raise ValueError(f'Unknown model_name: {args.model_name}')
