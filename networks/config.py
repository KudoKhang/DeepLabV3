from .libs import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="dataset/Figaro_1k/", help="Path to dataset")
    parser.add_argument("--num_classes", type=int, default=2, help="Num classes")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--pretrained", type=str, default='checkpoints/')
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    # print("           ⊱ ──────ஓ๑♡๑ஓ ────── ⊰")
    # print("🎵 hhey, arguments are here if you need to check 🎵")
    # for arg in vars(args):
    #     print("{:>15}: {:>30}".format(str(arg), str(getattr(args, arg))))
    # print()
    return args


