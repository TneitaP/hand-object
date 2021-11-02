# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime


def parse_dataset(args):
    """ Converts the --split argument into a dataset file """
    if args.split == 'aug':
        args.train_dataset = './data/perturbed_contactpose_train.pkl'
        args.test_dataset = './data/perturbed_contactpose_test.pkl'
    elif args.split == 'fine':
        args.test_dataset = './data/contactpose_test.pkl'
    elif args.split == 'im':
        args.test_dataset = './data/ho3d_image.pkl'
    elif args.split == 'demo':
        args.test_dataset = './data/ho3d_image_demo.pkl'
    else:
        raise ValueError('Unknown dataset')


def run_contactopt_parse_args():
    parser = argparse.ArgumentParser(description='Alignment networks training')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--split', default='aug', type=str,choices=['demo','aug','fine','im'])
    parser.add_argument('--lr', type=float)
    parser.add_argument('--n_iter', type=int)
    parser.add_argument('--partial', default=-1, type=int, help='Only run for n batches')
    parser.add_argument('--w_cont_hand', type=float, help='Weight of the hand contact in optimization')
    parser.add_argument('--sharpen_thresh', type=float)
    parser.add_argument('--ncomps', type=int)
    parser.add_argument('--w_cont_asym', type=float)
    parser.add_argument('--w_opt_trans', type=float)
    parser.add_argument('--w_opt_rot', type=float)
    parser.add_argument('--w_opt_pose', type=float)
    parser.add_argument('--caps_rad', type=float)
    parser.add_argument('--caps_hand', action='store_true')
    parser.add_argument('--cont_method', type=int)
    parser.add_argument('--caps_top', type=float)
    parser.add_argument('--caps_bot', type=float)
    parser.add_argument('--w_pen_cost', type=float)
    parser.add_argument('--pen_it', type=float)
    parser.add_argument('--w_obj_rot', type=float)
    parser.add_argument('--rand_re', type=int)
    parser.add_argument('--rand_re_trans', type=float)
    parser.add_argument('--rand_re_rot', type=float)
    parser.add_argument('--vis_method', type=int)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--min_cont', default=1, type=int, help='Cut grasps with less than this points of initial contact')
    args = parser.parse_args()
    parse_dataset(args)

    if args.vis:
        args.batch_size = 1

    return args

def run_contactopt_on_obman_parse_args():
    # Net argparse
    parser = argparse.ArgumentParser(description='Alignment networks training')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--n_iter', type=int)
    parser.add_argument('--partial', default=-1, type=int, help='Only run for n batches')
    parser.add_argument('--w_cont_hand', type=float, help='Weight of the hand contact in optimization')
    parser.add_argument('--sharpen_thresh', type=float)
    parser.add_argument('--ncomps', type=int)
    parser.add_argument('--w_cont_asym', type=float)
    parser.add_argument('--w_opt_trans', type=float)
    parser.add_argument('--w_opt_rot', type=float)
    parser.add_argument('--w_opt_pose', type=float)
    parser.add_argument('--caps_rad', type=float)
    parser.add_argument('--caps_hand', action='store_true')
    parser.add_argument('--cont_method', type=int)
    parser.add_argument('--caps_top', type=float)
    parser.add_argument('--caps_bot', type=float)
    parser.add_argument('--w_pen_cost', type=float)
    parser.add_argument('--pen_it', type=float)
    parser.add_argument('--w_obj_rot', type=float)
    parser.add_argument('--rand_re', type=int)
    parser.add_argument('--rand_re_trans', type=float)
    parser.add_argument('--rand_re_rot', type=float)
    parser.add_argument('--vis_method', type=int)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--min_cont', default=1, type=int, help='Cut grasps with less than this points of initial contact')

    # dataset argparse
    parser.add_argument('--root',default=R"F:\HOinter_data\Obman\downloads\obman", help="Path to dataset root")
    parser.add_argument(
        '--shapenet_root',default=R"F:\HOinter_data\ShapeNetCore\ShapeNetCore.v2", help="Path to root of ShapeNetCore.v2")
    parser.add_argument(
        '--split', type=str, default='test', help='Usually [train|test]')
    parser.add_argument(
        '--mode',
        default='all',
        choices=['all', 'obj', 'hand'],
        help='Mode for synthgrasp dataset')
    parser.add_argument(
        '--img_idx', type=int, default='1', help='Idx of first image to display')
    parser.add_argument(
        '--img_nb', type=int, default='10', help='Number of images to display')
    parser.add_argument(
        '--img_step', type=int, default='1', help='Number of images to display')
    parser.add_argument('--segment', action='store_true')
    parser.add_argument(
        '--mini_factor', type=float, help='Ratio in data to use (in ]0, 1[)')
    parser.add_argument(
        '--root_palm', action='store_true', help='Use palm as root')
    parser.add_argument('--use_cache', action='store_true', help='Use cache')
    parser.add_argument('--sides', type=str, default='left')
    parser.add_argument('--viz', default = True, action='store_true', help='Visualize samples')
    args = parser.parse_args()

    return args