import os
import multiprocessing
from itertools import groupby
import numpy as np
import pandas as pd
from scipy.linalg import solveh_banded
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from femstructure.utils.reader import read_tsv
from femstructure.utils.model_utils import parse_node

force_columns = ["Fx", "Fy", "Fz", "Mxx", "Myy", "Mzz"]
dir_columns = ["x", "y", "z", "xx", "yy", "zz"]


def plotter(forces, nodes, input_restraints, members, ntag2nidx):
    fig = plt.figure(figsize=(7,9))
    ax = fig.gca(projection='3d')
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 2, 1]))
    if isinstance(members, list):
        for m in members:
            n1, n2 = m['nodes']
            x, y, z = zip(*[nodes[n1], nodes[n2]])
            ax.plot(x, y, z, '-b')
    else:
        for mtag, m in members.items():
            n1, n2 = m['n1'], m['n2']
            x, y, z = zip(*[nodes[n1], nodes[n2]])
            ax.plot(x, y, z, '-b') 
    for label, restraints in input_restraints.items():
        restraints = [restraints[k] for k in dir_columns]
        xyz = nodes[label]
        rs = [int(r) for r in restraints[1:]]
        text = ""
        if sum(rs[:3]) > 0:
            text = 'o'
        if sum(rs[3:6]) > 0:
            text = '='
        if text:
            ax.text(xyz[0], xyz[1], xyz[2], text, fontsize='xx-large', color='black', ha='center', va='center_baseline')
    for ntag, d in forces.items():
        fs = [d[key] for key in force_columns]
        xyz = nodes[ntag]
        for f, dir in zip(fs, dir_columns):
            p = 0.0
            text = ""
            try:
                p = float(f)
                if p > 0.0:
                    text = '--->'
                elif p < 0.0:
                    text = '<---'
            except:
                print("ERROR: invalid acting forces:", f, fs)
            if text:
                if len(dir) > 1:
                    dir = dir[0]
                    text = text.replace('-->', 'O>>').replace('<--', '<<O')
                #print([p, text, dir])
                ax.text(xyz[0], xyz[1], xyz[2], text, fontsize='xx-large', weight='bold', color='red', zdir=dir)
    z_max = 0
    for label in ntag2nidx.keys():
        xyz = nodes[label]
        if xyz[2] > z_max:
            z_max = xyz[2]
        ax.text(xyz[0], xyz[1], xyz[2], label, fontsize='small')
    ax.set_xlabel('$X$', fontsize=14)
    ax.set_ylabel('$Y$', fontsize=14)
    ax.set_zlabel('$Z$', fontsize=14)
    ax.set_zlim(0, int(z_max*1.5))
    #ax.view_init(135, -110)
    plt.show()
    #plt.savefig('temp.png')


def plot_section(section_axis, input_data):
    def split_ntag(tag):
        tag_lst, start = [], 0
        for i, c in enumerate(tag):
            if not c.isdigit() and len(tag_lst) == 0:
                tag_lst += [tag[:i]]
                start = i
            elif c.isdigit() and len(tag_lst) == 1:
                tag_lst += [tag[start:i]]
                tag_lst += [tag[i:]]
                return tag_lst
        return []
    s_tag = section_axis.lower()
    pos = 1
    if s_tag.isdigit():
        pos = 2
    nodes = []
    for tag, d in input_data['nodes'].items():
        tag = tag.lower()
        lst = split_ntag(tag)
        if not lst:
            raise ValueError("ERROR: Invalid project data: node_tag is not valid: should be <level><axis_x><axis_y>, eg: 10B4")
        if lst[pos].lower() == s_tag:
            nodes += [(tag, [float(d[k]) for k in ['x', 'y', 'z']])]
    nodes = dict(nodes)
    members = []
    for tag, d in input_data['members'].items():
        n1, n2 = d['n1'].lower(), d['n2'].lower()
        if n1 in nodes and n2 in nodes:
            members += [(tag, n1, n2)]
    fig = plt.figure(figsize=(7, 10))
    fig.suptitle("section %s-%s"%(section_axis, section_axis))
    xs, ys = [], []          
    for tag, n1, n2 in members:
        x, y, z = zip(*[nodes[n1], nodes[n2]])
        if pos == 1:
            x = y
            y = z
        else:
            y = z
        xs += x
        ys += y
        plt.plot(x, y, '-b')
    to_origin_ntags = dict([(tag.lower(), tag) for tag in input_data['nodes'].keys()])
    for tag, (x, y, z) in nodes.items():
        if pos == 1:
            x = y
            y = z
        else:
            y = z
        plt.text(x, y, to_origin_ntags[tag], fontsize='small')
    for ntag, restraints in input_data['restraints'].items():
        ntag = ntag.lower()
        restraints = [restraints[k] for k in ['x','y', 'z', 'xx', 'yy', 'zz']]
        if ntag in nodes:
            x, y, z = nodes[ntag]
            rs = [int(r) for r in restraints]
            text = ""
            if sum(rs[:3]) > 0:
                text = 'o'
            if sum(rs[3:6]) > 0:
                text = '='
            if pos == 1:
                x = y
                y = z
            else:
                y = z
            if text:
                plt.text(x, y, text, fontsize='xx-large', color='black', ha='center', va='center_baseline')
    xs, ys = sorted(xs), sorted(ys)
    x_diff = (xs[-1] - xs[0])*0.2
    y_diff = (ys[-1] - ys[0])*0.2
    plt.xlim(xs[0]-x_diff, xs[-1]+x_diff)
    plt.ylim(ys[0]-y_diff, ys[-1]+y_diff)

    plt.show()


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--tsv_dir', dest='tsv_dir', type=str, default="", help="input tsv_dir or project")
    argparser.add_argument('-s', '--section_axes', dest='section_axes', type=str, default="A,1", help="view section: eg A")
    argparser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true', help="plot")
    args = argparser.parse_args()

    tsv_dir = ""
    if not os.path.isdir('_projects'):
        meta_dirs = []
    else:
        meta_dirs = os.listdir('_projects')
    if args.tsv_dir in meta_dirs:
        tsv_dir = os.path.join('_projects', args.tsv_dir, 'tsv')
    elif os.path.isdir(args.tsv_dir):
        tsv = args.tsv_dir
    else:
        raise ValueError("ERROR: Invalid input tsv_dir.")
    print("tsv_dir :%s"%tsv_dir)

    input_data = read_tsv(tsv_dir)
    nodes, _, ntag2nidx = parse_node(input_data['nodes'])
    #plotter(input_data['node_loadcases'], nodes, input_data['restraints'], input_data['members'], ntag2nidx)
    pros = []
    for section_axis in args.section_axes.split(','):
        pros += [multiprocessing.Process(target=plot_section, args=(section_axis, input_data))]
    for p in pros:
        p.start()

    for p in pros:
        p.join()
        p.close()


if __name__ == '__main__':
    main()
