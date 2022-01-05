import os
import sys
import json
import pandas as pd
from femstructure.utils.model_utils import wanted_keys
import argparse
current_dir = os.path.dirname(os.path.realpath(__file__))
defauft_config_path = os.path.join(current_dir, "config", "config_master.json")

def to_df(config):
    grid = config['grid']
    heights = config['height']
    df = {}
    nodes, ntags, restraints = [], [], []
    n = 0
    z = 0.0
    for i in range(len(heights)+1):
        l = str(i)
        x = 0.0
        for j, labx in enumerate(grid['label_x']):
            if j > 0:
                x += grid['length_x'][j-1]
            y = 0.0
            for k, laby in enumerate(grid['label_y']):
                if k > 0:
                    y += grid['length_y'][k-1]
                n += 1
                nodes += [(n, "%s%s%s"%(l, labx, laby), (x, y, z))]
                ntags == [("%s%s%s"%(l, labx, laby), n)]

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
                    m += 1
                    beam_x += [(m, "%sb%d"%(l,m), n1, n2, 'section_1')]
        for k, laby in enumerate(grid['label_y']):
            for j, labx in enumerate(grid['label_x']):
                if j > 0:
                    labx_1 = grid['label_x'][j-1]
                    n1, n2 = "%s%s%s"%(l, labx_1, laby), "%s%s%s"%(l, labx, laby)
                    m += 1
                    beam_y += [(m, "%sb%d"%(l,m), n1, n2, 'section_1')]
        l_1 = str(i - 1)
        for j, labx in enumerate(grid['label_x']):
            for k, laby in enumerate(grid['label_y']):
                n1, n2 = "%s%s%s"%(l_1, labx, laby), "%s%s%s"%(l, labx, laby)
                m += 1
                columns += [(m, "%sc%d"%(l,m), n1, n2, 'section_2')]
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
    return df

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config_path', dest='config_path', type=str, default=defauft_config_path, help="config_path")
    argparser.add_argument('-s', '--stories', dest='stories', type=int, default=-1, help="no of stories")
    argparser.add_argument('-o', '--output_tsv', dest='output_tsv', type=str, default="", help="tsv")
    argparser.add_argument('-v', '--verbosity', metavar="LEVEL", dest="verbosity", type=str, default='info',
                        help="Log verbosity. The config file is used if not given on the command line.")
    args = argparser.parse_args()

    with open(args.config_path, 'r') as fj:
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
        output_tsv = os.path.join('_meta', name, 'tsv')
    else:
        output_tsv = args.output_tsv
    print(output_tsv, os.path.isdir(output_tsv))
    if not os.path.isdir(output_tsv):
        os.system("mkdir -p %s"%output_tsv)

    df = to_df(config)
    for k, d in df.items():
        print(k, d.shape)
        print(d.head())
        d.to_csv(os.path.join(output_tsv, k+'.tsv'), sep='\t', index=False)
        print("save to: %s"%output_tsv)


if __name__ == '__main__':
    main()
