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
from femstructure.utils.model_utils import parse_node, get_max, to_string, plotter


def plot_thread(tsv_dir):
    input_data = read_tsv(tsv_dir)
    nodes, _, ntag2nidx = parse_node(input_data['nodes'])
    plotter(input_data['node_loadcases'], nodes, input_data['restraints'], input_data['members'], ntag2nidx)


def analysis_thread(tsv_dir, plot, job, verbose):
    if job == 'truss':
        f3d = Truss(tsv_dir)
    else:
        f3d = Frame(tsv_dir)
    #loadings = [('1', 0,0,-10000,0,0,0)]
    loadings = f3d.loadcases
    displacements, reactions = f3d(loadings)

    loading = f3d.get_tag(loadings, "loadings")
    displacement = f3d.get_tag(displacements, "displacements")
    reaction = f3d.get_tag(reactions, "reactions")
    forces = f3d.member_forces(displacements)
    if verbose:
        print("members:\n%s\n"%'\n'.join(["%-10s\t%s\t%s"%(m['tag'], "%s\t%s"%m['node_tags'], str(m['properties'])) for m in f3d.members]))
        for tag, values in [("loadings", loading), 
                            ("nodes", f3d.nodes.items()),
                            ("displacements", displacement),
                            ("reactions", reaction),
                            ("forces", forces)]:
            print("%s: %s"%(tag, to_string(values, job)))
    else:
        displacement_max = get_max(displacement)
        reaction_max = get_max(reaction)
        force_max = get_max(forces)
        for tag, values in [("loadings", loading), 
                            ("displacements_max", displacement_max),
                            ("reactions_max", reaction_max),
                            ("forces_max", force_max)]:
            print("%s: %s"%(tag, to_string(values, job)))  


def get_tsv_dir():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--input', dest='input_path', type=str, default='data/txt_examples/exB.3dd', help="input file(text) or project(name) or tsv_dir")
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

    return tsv_dir, args.job, args.plot, args.verbose


def frame_main():
    tsv_dir, _, plot, verbose = get_tsv_dir()
    if plot:
        p1 = multiprocessing.Process(target=plot_thread, args=(tsv_dir,))
    p2 = multiprocessing.Process(target=analysis_thread, args=(tsv_dir, plot, 'frame', verbose))
    if plot:
        p1.start()
    p2.start()
    #p1.join()
    p2.join()


def truss_main():
    tsv_dir, _, plot, verbose = get_tsv_dir()
    if plot:
        p1 =  multiprocessing.Process(target=plot_thread, args=(tsv_dir,))
    p2 =  multiprocessing.Process(target=analysis_thread, args=(tsv_dir, plot, 'truss', verbose))
    if plot:
        p1.start()
    p2.start()
    #p1.join()
    p2.join()


if __name__ == '__main__':
    frame_main()
