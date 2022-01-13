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

default_frame_input = 'data/txt_examples/exB.3dd'
default_truss_input = 'data/txt_examples/truss-p104.txt'


def plot_thread(project_dir, loading_dirs):
    input_data = read_tsv(project_dir, loading_dirs)
    nodes, _, ntag2nidx = parse_node(input_data['nodes'])
    node_loads = input_data['loadingcases'][loading_dirs[0]].get('node_loads', [])
    plotter(node_loads, nodes, input_data['restraints'], input_data['members'], ntag2nidx)


def run_case(f3d, output_dir, load_dir, verbose):
    displacements, reactions = f3d(loading_case=load_dir)
    forces = f3d.member_forces(displacements)
    loading_df = f3d.to_df(f3d.loadings_glob.get(load_dir, {}), "loadings")
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


def analysis_thread(project_dir, loading_dirs, plot, job, output_dir, verbose):
    if job == 'truss':
        f3d = Truss(project_dir, loading_dirs)
    else:
        f3d = Frame(project_dir, loading_dirs)
    #loadings = [('1', 0,0,-10000,0,0,0)]
    for load_dir in loading_dirs:
        print("running loading case: %s"%load_dir)
        run_case(f3d, output_dir, load_dir, verbose)


def get_project_dir(default_input_path=""):
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--input', dest='input_path', type=str, default=default_input_path, help="input file(text) or project(name) or project_dir")
    argparser.add_argument('-o', '--output_dir', dest='output_dir', type=str, default='', help="output_dir")
    argparser.add_argument('-l', '--loading_dirs', dest='loading_dirs', type=str, default="loadings", help="loadings: 'loadings,windload,deadload,liveload")
    argparser.add_argument('-j', '--job', dest='job', type=str, default="frame", help="job: frame/truss")
    argparser.add_argument('-p', '--plot', dest='plot', default=False, action='store_true', help="plot")
    argparser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true', help="verbose")
    args = argparser.parse_args()

    project_dir = ""
    if not os.path.isdir('_projects'):
        meta_dirs = []
    else:
        meta_dirs = os.listdir('_projects')
    if args.input_path in meta_dirs:
        project_dir = os.path.join('_projects', args.input_path)
    elif args.input_path.split('.')[0] in meta_dirs:
        project_dir = os.path.join('_projects', args.input_path.split('.')[0])
    elif args.input_path.endswith('.3dd') or args.input_path.endswith('.txt'):
        project_dir = os.path.join('_projects', args.input_path.split('/')[-1].split('.')[0])
        if os.path.isdir(project_dir):
            os.system("rm -r %s"%project_dir)
        project_dir = path_3dd_to_tsv_path(args.input_path)
        print("input_path :%s"%args.input_path)
    elif os.path.isdir(args.input_path):
        project_dir = args.input_path
    if not os.path.isdir(project_dir):
        raise ValueError("ERROR: Invalid input_path: %s"%args.input_path)
    
    args.project_dir = project_dir
    args.loading_dirs = args.loading_dirs.split(',')
    return args


def frame_main():
    args = get_project_dir(default_input_path=default_frame_input)
    project_dir, loading_dirs, plot, output_dir, verbose = args.project_dir, args.loading_dirs, args.plot, args.output_dir, args.verbose
    if plot:
        p1 = multiprocessing.Process(target=plot_thread, args=(project_dir, loading_dirs))
    p2 = multiprocessing.Process(target=analysis_thread, args=(project_dir, loading_dirs, plot, 'frame', output_dir, verbose))
    if plot:
        p1.start()
    p2.start()
    #p1.join()
    p2.join()


def truss_main():
    args = get_project_dir(default_input_path=default_truss_input)
    project_dir, loading_dirs, plot, output_dir, verbose = args.project_dir, args.loading_dirs, args.plot, args.output_dir, args.verbose
    if plot:
        p1 =  multiprocessing.Process(target=plot_thread, args=(project_dir, loading_dirs))
    p2 =  multiprocessing.Process(target=analysis_thread, args=(project_dir, loading_dirs, plot, 'truss', output_dir, verbose))
    if plot:
        p1.start()
    p2.start()
    #p1.join()
    p2.join()


if __name__ == '__main__':
    frame_main()
