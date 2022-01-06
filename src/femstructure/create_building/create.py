import os
import multiprocessing
import json
import pandas as pd
from femstructure.utils.model_utils import wanted_keys
from femstructure.main import plot_thread
import argparse
current_dir = os.path.dirname(os.path.realpath(__file__))
default_config_path = os.path.join(current_dir, "config_master.json")

def get_remove_members(remove_members):
    for mtag in remove_members:
        mtag = mtag.strip().lower()
        beam, (n1, n2) = mtag[0], tuple(mtag[1:].split('-'))
        yield "%s%s-%s"%(beam, n1, n2)
        yield "%s%s-%s"%(beam, n2, n1)


def to_df(config):
    grid = config['grid']
    heights = config['height']
    remove_node_tags = [x.lower() for x in config.get('remove_nodes', [])]
    remove_member_tags = list(get_remove_members(config.get('remove_members', [])))
    actual_remove_member_tags = []
    df = {}
    nodes, ntags, restraints = [], [], []
    n = 0
    z = 0.0
    for i in range(len(heights)+1):
        l = str(i)
        x = 0.0
        for j, labx in enumerate(grid['label_x']):
            labx = str(labx)
            if j > 0:
                x += grid['length_x'][j-1]
            y = 0.0
            for k, laby in enumerate(grid['label_y']):
                laby = str(laby)
                if k > 0:
                    y += grid['length_y'][k-1]
                ntag = "%s%s%s"%(l, labx, laby)
                if ntag.lower() not in remove_node_tags:
                    n += 1
                    nodes += [(n, ntag, (x, y, z))]
                    ntags += [(ntag, n)]

        z += heights[i-1]
        if i == 0:
            restraints = [x[:2] + tuple([1]*6) for x in nodes]
    members = []
    m = 0
    for i in range(len(heights)+1):
        if i == 0:
            continue
        l = str(i)
        beam_x, beam_y, columns = [], [], []
        for j, labx in enumerate(grid['label_x']):
            for k, laby in enumerate(grid['label_y']):
                if k > 0:
                    laby_1 = grid['label_y'][k-1]
                    n1, n2 = "%s%s%s"%(l, labx, laby_1), "%s%s%s"%(l, labx, laby)
                    if n1.lower() in remove_node_tags or n2.lower() in remove_node_tags:
                        continue
                    m += 1
                    #beam_x += [(m, "%sb%d"%(l,m), n1, n2, 'section_1')]
                    mtag = "b%s-%s"%(n1,n2)
                    if mtag.lower() in remove_member_tags:
                        actual_remove_member_tags += [mtag]
                        continue
                    beam_x += [(m, mtag, n1, n2, 'section_1')]
        for k, laby in enumerate(grid['label_y']):
            for j, labx in enumerate(grid['label_x']):
                if j > 0:
                    labx_1 = grid['label_x'][j-1]
                    n1, n2 = "%s%s%s"%(l, labx_1, laby), "%s%s%s"%(l, labx, laby)
                    if n1.lower() in remove_node_tags or n2.lower() in remove_node_tags:
                        continue
                    m += 1
                    mtag = "b%s-%s"%(n1,n2)
                    if mtag.lower() in remove_member_tags:
                        actual_remove_member_tags += [mtag]
                        continue
                    #beam_y += [(m, "%sb%d"%(l,m), n1, n2, 'section_1')]
                    beam_y += [(m, mtag, n1, n2, 'section_1')]
        l_1 = str(i - 1)
        for j, labx in enumerate(grid['label_x']):
            for k, laby in enumerate(grid['label_y']):
                n1, n2 = "%s%s%s"%(l_1, labx, laby), "%s%s%s"%(l, labx, laby)
                if n1.lower() in remove_node_tags or n2.lower() in remove_node_tags:
                    continue
                m += 1
                #columns += [(m, "%sc%d"%(l,m), n1, n2, 'section_2')]
                mtag = "c%s-%s"%(n1,n2)
                if mtag.lower() in remove_member_tags:
                    actual_remove_member_tags += [mtag]
                    continue
                columns += [(m, mtag, n1, n2, 'section_2')]
        members += beam_x + beam_y + columns
    data = [x[:2] + x[-1] for x in nodes]
    df['nodes'] = pd.DataFrame(data=data, columns=['nidx'] + wanted_keys['nodes'])
    df['members'] = pd.DataFrame(data=members, columns=['midx'] + wanted_keys['members'])
    df['restraints'] = pd.DataFrame(data=restraints, columns=['nidx'] + wanted_keys['restraints'])
    df['sections'] = pd.DataFrame(data=config['sections'], columns=wanted_keys['sections'])
    df['materials'] = pd.DataFrame(data=config['materials'], columns=wanted_keys['materials'])
    if 'node_loadcases' not in config:
        loadcases=[nodes[-1][:2]+(100.0, 100.0, 0.0, 0.0, 0.0, 0.0)]
    else:
        loadcases = [nodes[-1][:2] + tuple(config['node_loadcases'])]
    df['node_loadcases'] = pd.DataFrame(data=loadcases, columns=['nidx'] + wanted_keys['node_loadcases'])
    if actual_remove_member_tags:
        config.update({'remove_members': actual_remove_member_tags})


    return df


def remove_main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config_path', dest='config_path', type=str, default="", help="config_path")
    argparser.add_argument('-o', '--output_dir', dest='output_dir', type=str, default="", help="updated output dir")
    argparser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true', help="plot")
    args = argparser.parse_args()

    config_path = args.config_path
    if not args.config_path:
        config_path = default_config_path
    with open(config_path, 'r') as fj:
        config = json.load(fj)

    if not args.output_dir:
        raise ValueError("ERROR: no ouput dir.")
    if not args.config_path:
        print("default_config_path: %s"%config_path)
    else:
        print("input config_path  : %s"%config_path)
    print("output dir: %s"%args.output_dir)

    tsv_dir = os.path.join(args.output_dir, config['project'], 'tsv')
    if not os.path.isdir(tsv_dir):
        os.system("mkdir -p %s"%tsv_dir)

    df = to_df(config)
    for k, d in df.items():
        if args.verbose:
            print(k, d.shape)
            print(d.head())
        d.to_csv(os.path.join(tsv_dir, k+'.tsv'), sep='\t', index=False)
    p1 = multiprocessing.Process(target=plot_thread, args=(tsv_dir,))
    p1.start()
    remove_nodes_str = input("remove nodes?(eg. 5A1,4A1,3A1 or contine)\n>>")
    while remove_nodes_str[:1] != 'c':
        config.update({'remove_nodes': remove_nodes_str.split(',')})
        remove_members_str = input("remove member?(eg. b3H6-3H7,b3G7-3H7,c1H7-0H7 or contine)\n>>")
        if remove_members_str[:] == 'c':
            break
        config.update({'remove_members': remove_members_str.split(',')})
        df = to_df(config)
        for k, d in df.items():
            d.to_csv(os.path.join(tsv_dir, k+'.tsv'), sep='\t', index=False)
        p1.join()
        p1 = multiprocessing.Process(target=plot_thread, args=(tsv_dir,))
        p1.start()
        remove_nodes_str = input("remove nodes?(eg. 5A1,4A1,3A1 or contine)\n>>")
    p1.join()
    output_config_path = os.path.join(args.output_dir, "config_%s_updated.json"%config['project'])
    with open(output_config_path, 'w') as fj:
        fj.write(json.dumps(config, indent=2))
    print("save to: %s"%output_config_path)


def create_main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config_path', dest='config_path', type=str, default="", help="config_path")
    argparser.add_argument('-s', '--stories', dest='stories', type=int, default=-1, help="no of stories")
    argparser.add_argument('-o', '--output_tsv', dest='output_tsv', type=str, default="", help="tsv")
    argparser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true', help="plot")
    args = argparser.parse_args()

    config_path = args.config_path
    if not args.config_path:
        config_path = default_config_path
    with open(config_path, 'r') as fj:
        config = json.load(fj)
    if args.stories > 0:
        heights = config['height']
        len_heights = len(heights)
        if args.stories > len_heights:
            heights = heights + [heights[-1]]*(args.stories-len_heights)
        else:
            heights = heights[:args.stories]
        config.update({'height': heights})
    if not args.output_tsv:
        name = config['project']
        if args.stories > 0:
            name = "stories%d"%args.stories
            config.update({'project': name})
        output_tsv = os.path.join('_projects', name, 'tsv')
    else:
        output_tsv = args.output_tsv
    if args.verbose:
        print(output_tsv, os.path.isdir(output_tsv))
    if not os.path.isdir(output_tsv):
        os.system("mkdir -p %s"%output_tsv)

    df = to_df(config)
    for k, d in df.items():
        if args.verbose:
            print(k, d.shape)
            print(d.head())
        d.to_csv(os.path.join(output_tsv, k+'.tsv'), sep='\t', index=False)
    if not args.config_path:
        print("default_config_path: %s"%config_path)
    else:
        print("input config_path  : %s"%config_path)
    if args.stories > 0:
        if not os.path.isdir("output"):
            os.system("mkdir output")
        config_path = os.path.join("output", "config_stories%d.json"%args.stories)
        with open(config_path, 'w') as fj:
            json.dump(config, fj)
        print("save to            : %s"%config_path)
    print("save to            : %s"%output_tsv)
    if not args.config_path and args.stories < 0:
        print("try                :       create -s 15      (will get _projects/stories15/tsv/....)")

    plot_thread(output_tsv)


if __name__ == '__main__':
    create_main()
