import argparse

args = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="IRM Training")


    parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
    parser.add_argument("--weight_opt", help="Which optimizer to use for weight", default="sgd")

    parser.add_argument(
        "--log-dir", help="Where to save the runs. If None use ./runs", default=None
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=None,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )

    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )

    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )

    parser.add_argument(
        "--prune-rate",
        default=0.0,
        help="Amount of pruning to do during sparse training",
        type=float,
    )

    parser.add_argument(
        "--param-prune-rate",
        default=0.0,
        help="Amount of param pruning to do during sparse training",
        type=float,
    )


    parser.add_argument(
        "--random-subnet",
        action="store_true",
        help="Whether or not to use a random subnet when fine tuning for lottery experiments",
    )
    parser.add_argument(
        "--conv-type", type=str, default=None, help="What kind of sparsity to use"
    )
    parser.add_argument("--bn-type", default=None, help="BatchNorm type")
 


    parser.add_argument(
        "--weight_opt_lr",
        type=float,
        default=0.1,
        help="lr for weight training at the same time",
    )

    parser.add_argument('--use_cuda', default=True, action="store_true")

    parser.add_argument('--l2_regularizer_weight', type=float, default=0.00110794568)
    parser.add_argument('--algorithm',type=str ,default="irm" )
    parser.add_argument('--penalty_anneal_iters', type=int, default=100)
    parser.add_argument('--penalty_weight', type=float, default=4100.0)

    parser.add_argument('--encoder_size', type=int, default=64)
    parser.add_argument('--decoder_size', type=int, default=128)
    parser.add_argument('--in_length', type=int, default=10)
    parser.add_argument('--out_length', type=int, default=15)
    parser.add_argument('--grid_size', type=int, default=(13,13))
    parser.add_argument('--soc_conv_depth', type=int, default=64) 
    parser.add_argument('--conv_3x1_depth', type=int, default=16)
    parser.add_argument('--dyn_embedding_size', type=int, default=32)
    parser.add_argument('--input_embedding_size', type=int, default=32)
    parser.add_argument('--use_maneuvers',  default=False, action="store_true")
    parser.add_argument('--train_flag', default=True, action="store_true")
    parser.add_argument('--test_city', default="EP",type=str)
    parser.add_argument('--env_num', default=9,type=int)
    parser.add_argument('--train_data1', default="EP",type=str)
    parser.add_argument('--train_data2', default="EP",type=str)
    parser.add_argument('--test_data', default="OF",type=str)

    args = parser.parse_args()
    return args


def run_args():
    global args
    if args is None:
        print("init args...")
        args = parse_arguments()


run_args()
