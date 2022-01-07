#1.  python create_stories/create.py -c create_stories/config_two_story_frame.json
#2.  frame -i two_story_frame -p <to plot>
#3.  python test/structuralanalysis.py -i two_story_frame
#4.  python test/frame3dd.py -i two_story_frame
#
import os
import multiprocessing
import argparse
from femstructure.frame import Frame
from femstructure.truss import Truss
import numpy as np
from femstructure.utils.reader import path_3dd_to_tsv_path, read_tsv
from femstructure.utils.model_utils import parse_node, get_max_df, plotter


def plot_thread(tsv_dir):
    input_data = read_tsv(tsv_dir)
    nodes, _, ntag2nidx = parse_node(input_data['nodes'])
    plotter(input_data['node_loadcases'], nodes, input_data['restraints'], input_data['members'], ntag2nidx)


def analysis_thread(tsv_dir, plot, job, output_dir, verbose):
    if job == 'truss':
        f3d = Truss(tsv_dir)
    else:
        f3d = Frame(tsv_dir)
    #loadings = [('1', 0,0,-10000,0,0,0)]
    loadings = f3d.loadcases
    displacements, reactions = f3d(loadings)
    forces = f3d.member_forces(displacements)

    loading_df = f3d.to_df(loadings, "loadings")
    displacement_df = f3d.to_df(displacements, "displacements")
    reaction_df = f3d.to_df(reactions, "reactions")
    force_df = f3d.to_df(forces, "forces")
    if verbose:
        for tag, df in [("nodes", f3d.to_df(f3d.nodes, "nodes")),
                        ("restraints", f3d.to_df(f3d.input_data['restraints'],"restraints")),
                        ("loadings", loading_df), 
                        ("displacements", displacement_df),
                        ("reactions", reaction_df),
                        ("forces", force_df)]:
            print("%s:\n%s"%(tag, df.to_string(index=False)))
    else:
        displacement_max = get_max_df(displacement_df, f3d.job)
        reaction_max = get_max_df(reaction_df, f3d.job)
        force_max = get_max_df(force_df, f3d.job)
        for tag, df in [("loadings", loading_df), 
                        ("displacements_max", displacement_max),
                        ("reactions_max", reaction_max),
                        ("forces_max", force_max)]:
            print("%s:\n%s"%(tag, df))#.to_string(index=False)))  
    if output_dir:
        if not os.path.isdir(output_dir):
            os.system("mkdir -p %s"%output_dir)
        for tag, df in [("nodes", f3d.to_df(f3d.nodes, "nodes")),
                        ("restraints", f3d.to_df(f3d.input_data['restraints'],"restraints")),
                        ("loadings", loading_df), 
                        ("displacements", displacement_df),
                        ("reactions", reaction_df),
                        ("forces", force_df)]:
            path = os.path.join(output_dir, "%s.tsv"%tag)
            df.to_csv(path, sep='\t', index=False)
            print("save to: %s"%path)     


def get_tsv_dir():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--input', dest='input_path', type=str, default='data/txt_examples/exB.3dd', help="input file(text) or project(name) or tsv_dir")
    argparser.add_argument('-o', '--output_dir', dest='output_dir', type=str, default='', help="output_dir")
    argparser.add_argument('-j', '--job', dest='job', type=str, default="frame", help="job: frame/truss")
    argparser.add_argument('-p', '--plot', dest='plot', default=False, action='store_true', help="plot")
    argparser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true', help="verbose")
    args = argparser.parse_args()

    tsv_dir = ""
    if not os.path.isdir('_projects'):
        meta_dirs = []
    else:
        meta_dirs = os.listdir('_projects')
    if args.input_path in meta_dirs:
        tsv_dir = os.path.join('_projects', args.input_path, 'tsv')
        print("tsv_dir :%s"%tsv_dir)
    if args.input_path.endswith('.3dd') or args.input_path.endswith('.txt'):
        tsv_dir = path_3dd_to_tsv_path(args.input_path)
        print("input_path :%s"%args.input_path)
    elif args.input_path.endswith('tsv'):
        if os.path.isdir(args.input_path):
            tsv_dir = args.input_path
    if not tsv_dir:
        raise ValueError("ERROR: Invalid input_path: %s"%args.input_path)

    return tsv_dir, args.job, args.plot, args.output_dir, args.verbose


def frame_main():
    tsv_dir, _, plot, output_dir, verbose = get_tsv_dir()
    if plot:
        p1 = multiprocessing.Process(target=plot_thread, args=(tsv_dir,))
    p2 = multiprocessing.Process(target=analysis_thread, args=(tsv_dir, plot, 'frame', output_dir, verbose))
    if plot:
        p1.start()
    p2.start()
    #p1.join()
    p2.join()


def truss_main():
    tsv_dir, _, plot, output_dir, verbose = get_tsv_dir()
    if plot:
        p1 =  multiprocessing.Process(target=plot_thread, args=(tsv_dir,))
    p2 =  multiprocessing.Process(target=analysis_thread, args=(tsv_dir, plot, 'truss', output_dir, verbose))
    if plot:
        p1.start()
    p2.start()
    #p1.join()
    p2.join()


if __name__ == '__main__':
    frame_main()
