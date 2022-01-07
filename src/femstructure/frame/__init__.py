import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from numpy.linalg import inv
from ..utils.model_utils import (wanted_keys, parse_member_frame, parse_member_truss, parse_node, parse_restraint,
                        global_stiffness, fix_singularity, parse_loadcase, value_to_table, plotter)
from ..utils.reader import read_tsv


class Frame:
    def __init__(self, tsv_dir, job='frame'):
        self.job = job
        self.DOF = 6
        parse_member_func = parse_member_frame
        if job == 'truss':
            self.DOF = 3
            parse_member_func = parse_member_truss
        input_data = read_tsv(tsv_dir)
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
        self.loadcases = input_data['node_loadcases']
        self.input_data = input_data


    def fem_model(self):
        stiffness_matrix, bandwidth = global_stiffness(self.members, self.n_dof, self.DOF)
        sm = stiffness_matrix[np.ix_(self.didxs, self.didxs)]
        sm = fix_singularity(sm)
        #displacement_k = inv(sm)
        displacement_k = splu(csc_matrix(sm)) #lu
        reactions_k = stiffness_matrix[np.ix_(self.ridxs, self.didxs)]

        return stiffness_matrix, displacement_k, reactions_k


    def parse_loadcase(self, loadcase):

        return parse_loadcase(loadcase, self.ntag2nidx, self.didxs, self.n_dof, self.DOF)


    def to_df(self, values, where):
        if where == 'loadings':
            where = 'node_loadcases'
        keys = wanted_keys[where]
        rows = []
        if where == 'displacements':
            rows = list(value_to_table(self.didxs, values.reshape((1,-1))[0], self.nidx2ntag, self.DOF))
        elif where == 'reactions':
            rows = list(value_to_table(self.ridxs, values.reshape((1,-1))[0], self.nidx2ntag, self.DOF))
        elif where in ['node_loadcases', 'nodes', 'restraints']:
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


    def __call__(self, forces):
        if isinstance(forces, list):
            parsed_forces = []
            for fs in forces:
                if len(fs) < 7:
                    raise ValueError("ERROR: Invalid loading case: %s"%(str(fs)))
                fs = list(fs)
                parsed_forces += [(str(fs[0]), dict(list(zip(force_columns, fs[1:]))))]
            forces = dict(parsed_forces)
        forces = self.parse_loadcase(forces)
        #displacements = np.dot(self.displacement_k, forces)#;print(forces, displacements)
        displacements = self.displacement_k.solve(forces)
        reactions = np.dot(self.reactions_k, displacements)

        return displacements, reactions


    def member_forces(self, displacemets):
        d = [0.0]*self.DOF
        ds = dict([(ntag, d) for ntag in self.nidx2ntag.values()])
        rows = [(x[0], x[1:]) for x in value_to_table(self.didxs, displacemets.reshape((1,-1))[0], self.nidx2ntag, self.DOF)]
        ds.update(dict(rows))
        fs = []
        for m in self.members:
            n1, n2 = m['node_tags']
            d = np.concatenate((ds[n1], ds[n2]),axis=None).T#;print(m['tag']);print(np.dot(m['rotation_mat'], d));print(m['local_stiffness_mat'])
            fs += [(m['tag'], np.dot(m['local_stiffness_mat'], np.dot(m['rotation_mat'], d)))]

        return fs
    

    def plotter(self, forces=[]):
        plotter(forces, self.nodes, self.input_data['restraints'], self.members, self.ntag2nidx)