import os
import sys
import json
import numpy as np
import pandas as pd
from .model_utils import force_columns, dir_columns, wanted_keys
from .fixed_end_forces import  get_fixed_end_forces
wanted_keys = dict([(key, val) for key, val in wanted_keys.items() if key not in ['forces', 'reactions', 'displacements']])

#mapping from frame3dd input file xxx.3dd
tag_mapping = {
    "number of nodes": {"tag": "nodes", 
        "columns": ["node", "x", "y", "z", "r"], 
        "wanted_cols": ["node", "x", "y", "z"]},
    "number of nodes with reactions": {"tag": "restraints", 
        "columns": ["node", "x", "y", "z", "xx", "yy", "zz"], 
        "wanted_cols": ["node", "x", "y", "z", "xx", "yy", "zz"]},
    "number of nodes with restraints": {"tag": "restraints", 
        "columns": ["node", "x", "y", "z", "xx", "yy", "zz"], 
        "wanted_cols": ["node", "x", "y", "z", "xx", "yy", "zz"]},
    "number of frame elements": {"tag": "members",
        "columns": ["member", "n1", "n2", "Ax", "Asy", "Asz", "Jxx", "Iyy", "Izz", "E", "G", "roll", "density"],
        "wanted_cols": ["member", "n1", "n2", "Ax", "Asy", "Asz", "Jxx", "Iyy", "Izz"]},
    "number of members": {"tag": "members", 
        "columns": ["member", "n1", "n2", "Ax", "Asy", "Asz", "Jxx", "Iyy", "Izz", "E", "G", "roll", "density"],
        "wanted_cols": ["member", "n1", "n2", "Ax", "Asy", "Asz", "Jxx", "Iyy", "Izz"]},
    "number of materials": {"tag": "materials", "columns": ["material", "E", "G", "density"]},
    "number of loaded nodes": {"tag": "node_loads",  
        "columns": ["node", "Fx", "Fy", "Fz", "Mxx", "Myy", "Mzz"],
        "wanted_cols": ["node", "Fx", "Fy", "Fz", "Mxx", "Myy", "Mzz"]},
    "number of uniform distributed loads": {"tag": "uniformloads",  
        "columns": ["member", "Ux", "Uy", "Uz"],
        "wanted_cols": ["member", "Ux", "Uy", "Uz"]},
    "number of uniform loads": {"tag": "uniformloads",  
        "columns": ["member", "Ux", "Uy", "Uz"],
        "wanted_cols": ["member", "Ux", "Uy", "Uz"]},
    "number of internal concentrated loads": {"tag": "pointloads",  
        "columns": ["member", "Px", "Py", "Pz", "x"],
        "wanted_cols": ["member", "x", "Px", "Py", "Pz"]},
    "number of trapezoidally distributed loads": {"tag": "trapezoidalloads",  
        "columns": ["member", "xy1", "xy2", "wy1", "wy2", "xz1", "xz2", "wz1", "wz2"],
        "wanted_cols": ["member", "xy1", "xy2", "wy1", "wy2", "xz1", "xz2", "wz1", "wz2"]}}
base_keys = {'members', 'sections', 'restraints', 'materials', 'nodes'}


def extract_member(val):
    def match_section(f, sects):
        key = tuple(["%.6f"%float(x) for x in [f.Ax, f.Jxx, f.Iyy, f.Izz]])
        return sects.get(key, "")

    m = val['value'][0]
    E, G = m[9], m[10]
    sections = set([tuple(["%.6f"%float(row[i]) for i in [3,6,7,8]]) for row in val['value']])
    materials_df = pd.DataFrame(data=[("material_0", E, G)], columns=wanted_keys['materials'])
    df = pd.DataFrame(data = val['value'], columns=val['columns'])
    df = df[val['wanted_cols']]

    sects, data = {}, []
    for i, sect in enumerate(sections):
        stag = "section_%d"%i
        sects.update({sect: stag})
        data += [(stag,) + sect]
    sections_df = pd.DataFrame(data=data, columns=wanted_keys['sections'])
    df['section'] = df.apply(lambda row: match_section(row, sects), axis=1)

    return materials_df, sections_df, df[wanted_keys['members']]


def path_3dd_to_tsv_path(path):
    tags = [val['tag'] for val in tag_mapping.values()]
    tag_values = dict([(tag, {"value": [], "columns": [], "wanted_cols": []}) for tag in tags])
    tag, values, num_val, cols, wkeys = "", [], 0, [], []
    with open(path, 'r') as fd:
        lines = [line for line in fd if line.strip()]
    for line in lines:
        line = line.strip().replace('\t', ' ')
        if line[0] == '#':
            continue
        lst = line.split('#')
        if len(lst) > 1:
            num_str, tag_str = lst[0], lst[1]
            t = tag_mapping.get(tag_str.strip(), {})
            if t:
                tag, cols, wkeys = t["tag"], t["columns"], t['wanted_cols']
            if num_str:
                try:
                    num_val = int(num_str.strip())
                except:
                    vals = num_str.strip().split()
                    if vals[0].isdigit():
                        values += [vals]
                    if len(values) == num_val and tag and num_val > 0:
                        #print(tag, len(values), num_val, '-------1')
                        tag_values.update({tag: {"value": values, "columns": cols, 'wanted_cols': wkeys}})
                        tag, values, num_val, cols, wkeys = "", [], 0, [], []
                    continue
            if tag not in tag_values:
                num_val = 0
            if tag == 'trapezoidalloads':
                num_val = 3*num_val
            values = []
            continue
        elif len(lst) == 1:
            vals = line.split()
            if vals[0].isdigit():
                values += [vals]
            if len(values) == num_val and tag and num_val > 0:
                #print(tag, len(values), num_val, '-------2')
                tag_values.update({tag: {"value": values, "columns": cols, 'wanted_cols': wkeys}})
                tag, values, num_val, cols, wkeys = "", [], 0, [], []
        else:
            #print("WARNING: Invalid line %s"%str([line]), num_val, tag)
            pass
        
    data_df = {}
    for key, val in tag_values.items():
        if key == 'materials':
            continue
        if key == "members":
            materials_df, sections_df, df = extract_member(val)
            data_df.update({"members": df, "materials": materials_df, "sections": sections_df})
        elif key in ['trapezoidalloads'] and val['value']:
            lst = val['value']
            k, val_lst = len(lst[0]), []
            for i in range(0, len(lst), 3):
                row = lst[i]
                if len(row) == k:
                    val_lst += [row[:1] + lst[i+1]+lst[i+2]]
            df = pd.DataFrame(data = val_lst, columns=val['columns'])
            df = df[val['wanted_cols']]
            data_df.update({key: df[wanted_keys[key]]})
        elif val['columns']:
            df = pd.DataFrame(data = val['value'], columns=val['columns'])
            df = df[val['wanted_cols']]
            data_df.update({key: df[wanted_keys[key]]})
    assert(len(base_keys - set(data_df.keys()))==0), (base_keys, set(data_df.keys()))
    project_dir = os.path.join("_projects", path.split('/')[-1].split('.')[0])
    tsv_dir = os.path.join(project_dir, 'tsv')
    loads_dir = os.path.join(project_dir, 'loadings')
    for _dir in [tsv_dir, loads_dir]:
        if not os.path.isdir(_dir):
            os.system("mkdir -p %s"%_dir)
    member_loading_dfs = []
    for key, df in data_df.items():
        if key.endswith('loads'):
            path = os.path.join(loads_dir, key+'.tsv')
            if key not in ['node_loads']:
                member_loading_dfs += [(key, df)]
        else:
            path = os.path.join(tsv_dir, key+'.tsv')
        df.to_csv(path, sep='\t', index=False)
        print("save to: %s"%path)
    df = get_fixed_end_forces(member_loading_dfs, data_df['nodes'], data_df['members'])
    path = os.path.join(loads_dir, "fixed_end_forces.tsv")
    df.to_csv(path, sep='\t', index=False)
    print("save to: %s"%path)
    return project_dir


def parse_dir(_dir):
    data = []
    for f in os.listdir(_dir):
        if not f.endswith('.tsv'):
            continue
        key = f.split('.')[0]
        if key not in wanted_keys.keys():
            raise ValueError("WARNING: Invalid tsv data file in %s"%str((_dir, key, f)))
        df = pd.read_csv(os.path.join(_dir, f), sep='\t', dtype=str)
        df = df[wanted_keys[key]]
        #print(key, df.shape);print(df.head())
        cols = df.columns
        ks = df[cols[0]].tolist()
        val_df = df[cols[1:]]
        dict_to_lst = dict(zip(ks, val_df.values.tolist())) #dict(zip(ks, val_df.T.to_dict().values()))
        data += [(key, dict_to_lst)]
    return data


def read_tsv(project_dir, loading_dirs):
    data, loadingcases = [], {}
    for tag in ['tsv'] + loading_dirs:
        _dir = os.path.join(project_dir, tag)
        if tag == 'tsv':
            geometry_data = parse_dir(_dir)
            data += geometry_data
        else:
            if os.path.isdir(_dir):
                load_dict = dict(parse_dir(_dir))
                keys = set(load_dict.keys())
                if "fixed_end_forces" not in keys and len(keys - set(['node_loads'])) > 0:
                    tsv_data = dict(geometry_data)
                    enfs = fixed_end_forces(load_dict, tsv_data['nodes'], tsv_data['members'], _dir)
                    load_dict.update({"fixed_end_forces": enfs})
                loadingcases.update({tag: load_dict})
    data += [("loadingcases", loadingcases)]
    return dict(data)


def fixed_end_forces(member_loading_dict, nodes, members, loads_dir):
    dfs = []
    for key, data in member_loading_dict.items():
        cols = wanted_keys[key]
        lst = [(k,) + tuple([d[k] for k in cols[1:]]) for k, d in data.items()]
        dfs += [(key, pd.DataFrame(data=lst, columns = cols))]
    cols = wanted_keys['nodes']
    node_df = pd.DataFrame(data=[(k,) + tuple([d[k] for k in cols[1:]]) for k, d in nodes.items()], columns=cols)
    cols = wanted_keys['members']
    member_df = pd.DataFrame(data=[(k,) + tuple([d[k] for k in cols[1:]]) for k, d in members.items()], columns=cols)
    df = get_fixed_end_forces(dfs, node_df, member_df)
    path = os.path.join(loads_dir, "fixed_end_forces.tsv")
    df.to_csv(path, sep='\t', index=False)
    print("save to: %s"%path)

    cols = wanted_keys['fixed_end_forces']
    enfs = []
    for row in df.values.tolist():
        enfs += [(row[0], dict([(k, v) for k, v in zip(cols[1:], row[1:])]))]
    return enfs