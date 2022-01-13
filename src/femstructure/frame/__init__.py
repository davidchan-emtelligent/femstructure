import numpy as np
import pandas as pd
from itertools import groupby
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from numpy.linalg import inv
from ..utils.model_utils import (wanted_keys, parse_member_frame, parse_member_truss, parse_node, parse_restraint,
                        global_stiffness, fix_singularity, parse_loadcase, value_to_table, plotter)
from ..utils.reader import read_tsv

def loadings_on_nodes(loadingcases, members):
    mtag_to_data = []
    for m in members:
        mtag_to_data += [(m['tag'], {'node_tags': m['node_tags'], 'rotation_mat': m['rotation_mat']})]
    mtag_to_data = dict(mtag_to_data)
    def to_node_forces(fixed_end_forces):
        for mtag, fs in fixed_end_forces.items():
            m = mtag_to_data[mtag]
            n1, n2 = m['node_tags']
            ett = np.transpose(m['rotation_mat'])
            fs = -1*np.array(fs, dtype=np.float64).reshape((-1,1))#;print(ett, fs)
            fs = list(np.dot(ett, fs).flatten())#;print(fs, [n1])
            yield n1, fs[:6]
            yield n2, fs[6:]
    ret = []
    for key, loadings in loadingcases.items():
        node_loads = loadings.get('node_loads', {}).items()
        if len(node_loads) > 0:
            node_loads = [(ntag, [float(v) for v in row]) for ntag, row in node_loads]
        else:
            node_loads = []
        fixed_end_forces = list(to_node_forces(loadings.get('fixed_end_forces', {})))
        n_loads = []
        if fixed_end_forces:
            for tag, g in groupby(sorted(fixed_end_forces + node_loads, key=lambda x: x[0]), key=lambda x: x[0]):
                lst = [row for _, row in g]
                row = np.sum(np.array(lst), axis=0)
                n_loads += [(tag,) + tuple(row)]
        else:
            n_loads = [(tag,) + tuple(row) for tag, row in node_loads]
        ret += [(key, n_loads)]#;print(n_loads)

    return dict(ret)


class Frame:
    def __init__(self, project_dir, loading_dirs, job='frame'):
        self.job = job
        self.DOF = 6
        parse_member_func = parse_member_frame
        if job == 'truss':
            self.DOF = 3
            parse_member_func = parse_member_truss
        input_data = read_tsv(project_dir, loading_dirs)
        #nodes
        self.nodes, self.nidx2ntag, self.ntag2nidx = parse_node(input_data['nodes'])
        self.n_dof = len(self.nodes)*self.DOF
        #supports
        self.ridxs = parse_restraint(input_data['restraints'], self.ntag2nidx, self.DOF)
        self.didxs = np.setdiff1d(np.arange(self.n_dof), self.ridxs)
        #members
        self.members = list(parse_member_func(input_data['members'], self.nodes, self.ntag2nidx, input_data['materials'], input_data['sections']))
        #molels: matrixes
        self.stiffness_matrix, self.displacement_k, self.reactions_k = self.fem_model()
        self.loadings_glob = loadings_on_nodes(input_data['loadingcases'], self.members)
        self.input_data = input_data
        self.loadingcase = "" #call() will provide the current case

    def fem_model(self):
        stiffness_matrix, bandwidth = global_stiffness(self.members, self.n_dof, self.DOF)
        sm = stiffness_matrix[np.ix_(self.didxs, self.didxs)]
        sm = fix_singularity(sm)
        """
        print(self.didxs)
        for x in [sm]:
            print('-----------')
            for vs in x:
                print('\t'.join(['%10.2f'%(v) for v in vs]))
        #"""
        #displacement_k = inv(sm)
        displacement_k = splu(csc_matrix(sm)) #lu
        reactions_k = stiffness_matrix[np.ix_(self.ridxs, self.didxs)]

        return stiffness_matrix, displacement_k, reactions_k


    def parse_loadcase(self, loadcase):

        return parse_loadcase(loadcase, self.ntag2nidx, self.didxs, self.n_dof, self.DOF)


    def to_df(self, values, where):
        if where == 'loadings':
            where = 'node_loads'
        keys = wanted_keys[where]
        rows = []
        if where == 'displacements':
            rows = list(value_to_table(self.didxs, values.reshape((1,-1))[0], self.nidx2ntag, self.DOF))
        elif where == 'reactions':
            rows = list(value_to_table(self.ridxs, values.reshape((1,-1))[0], self.nidx2ntag, self.DOF))
            ntag2forces = dict([(row[0], list(row[1:1+self.DOF])) for row in self.loadings_glob[self.loadingcase]])#all ntag is reactions from fixed_end_forces
            ret = []
            for row in rows:
                fixed_end_forces = ntag2forces.get(row[0], [])
                if fixed_end_forces:
                    row = (row[0],) + tuple(np.sum([np.array(row[1:]), -1*np.array(fixed_end_forces)], axis=0))
                ret += [row]
            rows = ret
        elif where in ['node_loads', 'nodes', 'restraints']:
            kkeys = keys[1:]
            if isinstance(values, dict):
                for ntag, d in values.items():
                    if isinstance(d, dict):
                        rows += [[ntag] + [d[key] for key in kkeys]]
                    else:
                        rows += [[ntag] + list(d)]
            else:
                for row in values:
                    rows += [[row[0]] + [float(v) for  v in row[1:]]]
        elif where == 'forces':
            for tag, value in values:
                rows += [[tag] + list(value)]
            if self.job == 'truss' and len(keys) > 9:
                keys = keys[:-9] + keys[-6:-3]
        else:
            raise ValueError("ERROR: Invalid key in to_df(): %s"%where)

        return pd.DataFrame(data=rows, columns=keys[:len(rows[0])])


    def member_forces(self, displacements):
        fixed_end_forces = self.input_data['loadingcases'][self.loadingcase].get('fixed_end_forces', {})
        tups = []
        for mtag, f in fixed_end_forces.items():
            tups += [(mtag, [float(v) for v in f])]
        mtag2fs = dict(tups)
        d = [0.0]*self.DOF
        ds = dict([(ntag, d) for ntag in self.nidx2ntag.values()])
        rows = [(x[0], x[1:]) for x in value_to_table(self.didxs, displacements.reshape((1,-1))[0], self.nidx2ntag, self.DOF)]
        ds.update(dict(rows))
        fs = []
        for m in self.members:
            mtag = m['tag']
            n1, n2 = m['node_tags']
            d = np.concatenate((ds[n1], ds[n2]),axis=None).T#;print(m['tag']);print(np.dot(m['rotation_mat'], d));print(m['local_stiffness_mat'])
            f = np.dot(m['local_stiffness_mat'], np.dot(m['rotation_mat'], d))
            fixed_end_f = mtag2fs.get(mtag, [])
            if fixed_end_f:
                f = np.sum([f, np.array(fixed_end_f)], axis=0)
            fs += [(mtag, f)]

        return fs
    

    def __call__(self, forces=tuple([]), loading_case='loadings'):
        self.loadingcase = loading_case
        if forces:
            forces = list(forces)
        else:
            forces = self.loadings_glob.get(loading_case, {})
            if not forces:
                raise ValueError("ERROR: no loading: %s"%loading_case)
        if isinstance(forces, list):
            parsed_forces = []
            for fs in forces:
                if len(fs) < 7:
                    raise ValueError("ERROR: Invalid loading case: %s"%(str(fs)))
                fs = list(fs)
                parsed_forces += [(str(fs[0]), dict(list(zip(wanted_keys["node_loads"][1:], fs[1:]))))]
            forces = dict(parsed_forces)
        forces = self.parse_loadcase(forces)
        displacements = self.displacement_k.solve(forces)#displacements = np.dot(self.displacement_k, forces), displacement_k = lu
        reactions = np.dot(self.reactions_k, displacements)

        return displacements, reactions

        
    def plotter(self, forces=[]):
        plotter(forces, self.nodes, self.input_data['restraints'], self.members, self.ntag2nidx)