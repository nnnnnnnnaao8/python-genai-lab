# echo_args.py
#import argparse
#ap = argparse.ArgumentParser()
#ap.add_argument("--in",  dest="in_path")
#ap.add_argument("--out", dest="out_path")
#ap.add_argument("--fig", dest="fig_path")
#ap.add_argument("--scale", action="store_true")
#args = ap.parse_args()
#print(vars(args))          # ← 中身をそのまま辞書で表示

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--head", type=int, default=None)
args = ap.parse_args()
print("HEAD =", args.head)
