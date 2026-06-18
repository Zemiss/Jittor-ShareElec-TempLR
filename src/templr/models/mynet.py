from jittor_geometric.nn.models.craft import CRAFT


class MyNetNetwork(CRAFT):
    """Independent CRAFT copy for experiments without touching baseline."""

    def __init__(self, args, node_size):
        super().__init__(
            n_layers=2,
            n_heads=2,
            hidden_size=args.hidden_size,
            hidden_dropout_prob=args.dropout,
            attn_dropout_prob=args.dropout,
            hidden_act='gelu',
            layer_norm_eps=1e-12,
            initializer_range=0.02,
            n_nodes=node_size,
            max_seq_length=args.num_neighbors,
            loss_type='BPR',
            use_pos=True,
            input_cat_time_intervals=False,
            output_cat_time_intervals=True,
            output_cat_repeat_times=True,
            num_output_layer=1,
            emb_dropout_prob=args.dropout,
            skip_connection=True,
        )


def build_mynet_model(args, node_size):
    return MyNetNetwork(args, node_size)
