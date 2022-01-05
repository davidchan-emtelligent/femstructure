import os
import sys
import json
import numpy as np
import pandas as pd
from .model_utils import force_columns, dir_columns, wanted_keys


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
    "number of loaded nodes": {"tag": "node_loadcases",  
        "columns": ["node", "Fx", "Fy", "Fz", "Mxx", "Myy", "Mzz"],
        "wanted_cols": ["node", "Fx", "Fy", "Fz", "Mxx", "Myy", "Mzz"]},
    "number of materials": {"tag": "materials", "columns": ["material", "E", "G", "density"]}}


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
        lst = line.split('#')
        if len(lst) > 1 and lst[0].strip():
            if len(values) == num_val and tag and num_val > 0:
                tag_values.update({tag: {"value": values, "columns": cols, 'wanted_cols': wkeys}})
            tag, values, num_val, cols, wkeys = "", [], 0, [], []
            num_str, tag_str = lst[0], lst[1]
            t = tag_mapping.get(tag_str.strip(), {})
            if t:
                tag, cols, wkeys = t["tag"], t["columns"], t['wanted_cols']
            if num_str:
                try:
                    num_val = int(num_str.strip())
                except:
                    #print('WARNING: %s'%str([num_str, line]))
                    continue
            if tag not in tag_values:
                num_val = 0
        elif len(lst) == 1:
            values += [line.split()]
        else:
            #print("WARNING: Invalid line %s"%str([line]), num_val, tag)
            pass
    if len(values) == num_val and tag and num_val > 0:
        tag_values.update({tag: {"value": values, "columns": cols, 'wanted_cols': wkeys}})
    data_df = {}
    for key, val in tag_values.items():
        if key == 'materials':
            continue
        if key == "members":
            materials_df, sections_df, df = extract_member(val)
            data_df.update({"members": df, "materials": materials_df, "sections": sections_df})
        else:
            df = pd.DataFrame(data = val['value'], columns=val['columns'])
            df = df[val['wanted_cols']]
            data_df.update({key: df[wanted_keys[key]]})
    assert(set(data_df.keys())==set(wanted_keys.keys())), (set(data_df.keys()), set(wanted_keys.keys()))
    out_dir = os.path.join("_meta", path.split('/')[-1].split('.')[0], 'tsv')
    if not os.path.isdir(out_dir):
        os.system("mkdir -p %s"%out_dir)
    for key, df in data_df.items():
        df.to_csv(os.path.join(out_dir, key + '.tsv'), sep='\t', index=False)
        print("save to: %s"%(os.path.join(out_dir, key + '.tsv')))

    return out_dir


def read_tsv(path_dir):
    data = []
    for key in wanted_keys.keys():
        df = pd.read_csv(os.path.join(path_dir, key + '.tsv'), sep='\t', dtype=str)
        df = df[wanted_keys[key]]
        #print(key, df.shape);print(df.head())
        cols = df.columns
        ks = df[cols[0]].tolist()
        val_df = df[cols[1:]]
        dict_to_lst = dict(zip(ks, val_df.T.to_dict().values()))
        data += [(key, dict_to_lst)]

    return dict(data)
