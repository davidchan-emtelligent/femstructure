# Import Instruction for select local y' axis:
# The y' axis is selected to be perpendicular to the x' and Z axes
# in such a way that the cross product of global Z with x' results in the y' axis,
# thus (y' = Z * x').
#
# Select of y' for each element:
# Ensure global y is up, x is face to right and z is coming out
# Horizontal element : x' facing to right if possible -> y' is up
#                      x' facing to left -> y' is down
# vertical elements  : x' facing to upward -> y' to right
#                      x' facing to downward -> y' to left
#
from itertools import groupby
import numpy as np
from scipy.linalg import solveh_banded
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

verbose = False

force_columns = ["Fx", "Fy", "Fz", "Mxx", "Myy", "Mzz"]
dir_columns = ["x", "y", "z", "xx", "yy", "zz"]
wanted_keys = {
    "nodes": ["node", "x", "y", "z"],
    "restraints": ["node", "x", "y", "z", "xx", "yy", "zz"],
    "members": ["member", "n1", "n2", "section"],
    "node_loadcases": ["node", "Fx", "Fy", "Fz", "Mxx", "Myy", "Mzz"],
    "sections": ["section", "Ax", "Jxx", "Iyy", "Izz"],
    "materials": ["material", "E", "G"]
}
#    "sections": ["section", "Ax", "Jxx", "Iyy", "Izz"],

#nodes parsing:
def parse_node(node_data):
    nodes = {}
    didx_ntag = []
    ntag_didx = []
    for i, (ntag, d) in enumerate(node_data.items()):
        x, y, z = d['x'], d['y'], d['z']
        nodes[ntag] = (float(x), float(y), float(z))
        didx_ntag += [(i, ntag)]
        ntag_didx += [(ntag, i)]

    return nodes, dict(didx_ntag), dict(ntag_didx)


def parse_restraint(restraint_data, ntag2nidx, _dof):
    restraints = []
    for ntag, d in restraint_data.items():
        node_idx = ntag2nidx[ntag]
        rs = [d[k] for k in dir_columns[:_dof]]
        base = node_idx*_dof
        for i, r in enumerate(rs):
            if r == '1':
                restraints += [base]
            base += 1

    return sorted(restraints)


#loadcase parsing:
def parse_loadcase(loadcase, ntag2nidx, didxs, n_dof, _dof):
    forces = np.zeros((n_dof), dtype=np.double)
    for ntag, d in loadcase.items():
        fs = [d[key] for key in force_columns]
        for i, f in enumerate(fs):
            f = float(f)
            if f != 0.0:
                didx = ntag_dir_to_didx(ntag, i, _dof, ntag2nidx)
                forces[didx] = f

    return forces[didxs].reshape((-1, 1))


#rotation parsing: ref:p279 - Assumption: y' is perpendicular to Zx' plane, thus y' = Z*x'
def _rotation_3_3(mtag, l, m, n):
    D = (l*l + m*m)**0.5
    # ref:p281: special case : Z and x' are coinside/opposite
    if D < 0.001:
        if verbose:
            print("WARNING:",[l, m, n], "m%s aligned on z axis!"%mtag)

        return np.array([[0.0, 0.0, n],
                [0.0, 1, 0.0],
                [-n, 0.0, 0.0]])
    #ref:p280
    return np.array([[l, m, n],
            [-m/D, l/D, 0.0],
            [-l*n/D, -m*n/D, D]])


#solve banded matrix
def bm_inv(A, x, banded=True):
    if not banded or x.size() == 0:
        A = A.toarray()
        N = np.shape(A)[0]
        D = np.count_nonzero(A[0,:])
        ab = np.zeros((D,N))
        for i in np.arange(1,D):
            ab[i,:] = np.concatenate((np.diag(A,k=i),np.zeros(i,)),axis=None)
        ab[0,:] = np.diag(A,k=0)
        if x.size() == 0:

            return ab
    else:
        ab = A
    y = solveh_banded(ab,x,lower=True)
    
    return y


def _rotation(mtag, n1, n2, dim=12):
    dx, dy, dz = (n2[0] - n1[0]), (n2[1] - n1[1]), (n2[2] - n1[2])
    length = (dx*dx +dy*dy +dz*dz)**0.5
    l, m, n = float(dx)/length, float(dy)/length, float(dz)/length
    rotate_mat = _rotation_3_3(mtag, l, m, n)
    
    et = np.zeros((dim, dim), dtype=np.double)
    for i in range(0, dim, 3):
        et[i:i+3, i:i+3] = rotate_mat
        
    return et, (l, m, n, length)


def _stiffness_frame_local(l, A, J, Iy, Iz, E, G):
    invl2 = 1.0/l/l
    invl3 = invl2/l
    ae = A*E/l
    z12 = 12*E*Iz*invl3
    z6 = 6*E*Iz*invl2
    z4 = 4*E*Iz/l
    z2 = 2*E*Iz/l
    y12 = 12*E*Iy*invl3
    y6 = 6*E*Iy*invl2
    y4 = 4*E*Iy/l
    y2 = 2*E*Iy/l
    gj = G*J/l

    return np.array([[ae, 0, 0 , 0, 0, 0, -ae, 0, 0, 0, 0, 0],
                [0, z12, 0, 0, 0, z6, 0, -z12, 0, 0, 0, z6],
                [0, 0, y12, 0, -y6, 0, 0, 0, -y12, 0, -y6, 0],
                [0, 0, 0, gj, 0, 0, 0, 0, 0, -gj, 0, 0],
                [0, 0, -y6, 0, y4, 0, 0, 0, y6, 0, y2, 0],
                [0, z6, 0, 0, 0, z4, 0, -z6, 0, 0, 0, z2],
                [-ae, 0, 0, 0, 0, 0, ae, 0, 0, 0, 0, 0],
                [0, -z12, 0, 0, 0, -z6, 0, z12, 0, 0, 0, -z6],
                [0, 0, -y12, 0, y6, 0, 0, 0, y12, 0, y6, 0],
                [0, 0, 0, -gj, 0, 0, 0, 0, 0, gj, 0, 0],
                [0, 0, -y6, 0, y2, 0, 0, 0, y6, 0, y4, 0],
                [0, z6, 0, 0, 0, z2, 0, -z6, 0, 0, 0, z4]], dtype=np.double)


def _stiffness_truss_global(paras, A, E):
    (cx, cy, cz, length) = paras
    k = np.array([[cx*cx, cx*cy, cx*cz],
                [cx*cy, cy*cy, cy*cz],
                [cx*cz, cy*cz, cz*cz]])
    k1 = np.concatenate((k, -1.0*k), axis=0)
    k2 = np.concatenate((-1.0*k, k), axis=0)

    return (E*A/length)* np.concatenate((k1, k2), axis=1)


def _stiffness_frame_global(et, ek):

    return np.dot(np.transpose(et), np.dot(ek, et))
    #return np.dot(inv(et), np.dot(ek, et))


def parse_member_frame(member_data, nodes, ntag2nidx, materials, sections):
    material = list(materials.values())[0]
    E, G = float(material['E']), float(material['G'])
    for mtag, d in member_data.items():
        n1, n2, stag = d['n1'], d['n2'], d['section']
        section = sections[stag]
        A, Ix, Iy, Iz = float(section['Ax']), float(section['Jxx']), float(section['Iyy']), float(section['Izz'])
        member = {}
        member['tag'], member['node_tags'], member['properties'] = mtag, (n1, n2), (A, Ix, Iy, Iz, E, G)
        et, (l, m, n, length) = _rotation(mtag, nodes[n1], nodes[n2], dim=12)
        ek = _stiffness_frame_local(length, A, Ix, Iy, Iz, E, G)
        member['idxs'], member['nodes'], member['length'] = (ntag2nidx[n1], ntag2nidx[n2]), (n1, n2), length
        gk = _stiffness_frame_global(et, ek)
        member['rotation_mat'], member['local_stiffness_mat'], member['global_stiffness_mat'] = et, ek, gk
        """
        print(mtag)
        for x in [et, ek, gk]:
            print('-----------')
            for vs in x:
                print('\t'.join(['%10.2f'%(v) for v in vs]))
        #"""
        yield member


def parse_member_truss(member_data, nodes, ntag2nidx, materials, sections):
    material = list(materials.values())[0]
    E = float(material['E'])
    for mtag, d in member_data.items():
        n1, n2, stag = d['n1'], d['n2'], d['section']
        section = sections[stag]
        A = float(section['Ax'])
        member = {}
        member['tag'], member['node_tags'], member['properties'] = mtag, (n1, n2), (A, E)
        et, (cx, cy, cz, length) = _rotation(mtag, nodes[n1], nodes[n2], dim=6)
        gk = _stiffness_truss_global((cx, cy, cz, length), A, E)
        ae = (A*E/length)
        ek = np.array([[ae, 0, 0, -ae, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [-ae, 0, 0, ae, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]])
        member['idxs'], member['nodes'], member['length'] = (ntag2nidx[n1], ntag2nidx[n2]), (n1, n2), length
        member['rotation_mat'], member['local_stiffness_mat'], member['global_stiffness_mat'] = et, ek, gk
        """
        print(mtag,(cx, cy, cz, length));print(et)
        xv = length/A/E
        for vs in gk:
            print('\t'.join(['%10.2f'%(v*xv) for v in vs]))
        #"""
        yield member


def parse_member_fd_truss(member_data, nodes, ntag2nidx, materials):
	for m in member_data:
		(mtag, n1, n2, A) = tuple(m[:4])
		if len(m) > 9:
			E_m, G_m = m[9], m[10]
		if materials:
			E, G = materials['E'], materials['G']
		else:
			E, G = float(E_m), float(G_m)
		A = float(A)
		member = {}
		member['tag'], member['node_tags'], member['properties'] = mtag, (n1, n2), A
		et, (cx, cy, cz, length) = _rotation(mtag, nodes[n1], nodes[n2], dim=3)
		R = np.array([[cx*cx, cx*cy, cx*cz],
					  [cx*cy, cy*cy, cy*cz],
					  [cx*cz, cy*cz, cz*cz]])
		C = (E*A/length)*R
		member['idxs'], member['length'] = (ntag2nidx[n1], ntag2nidx[n2]), length
		member['global_rotation_mat'], member['global_characterist_mat'] = R, C

		yield member


#fem solver:
def global_stiffness(members, n_dof, _dof):
    global_k = np.zeros((n_dof,n_dof), dtype=np.double)
    max_band = 0
    for m in members:
        (i, j) = m['idxs']
        idxs = list(range(i*_dof, (i + 1)*_dof)) + list(range(j*_dof, (j + 1)*_dof))
        global_k[np.ix_(idxs,idxs)] += m['global_stiffness_mat']
        band = abs(j - i) + 1
        if band > max_band:
            max_band = band

    return global_k, max_band*_dof


def fix_singularity(sm):
    zero_rows = []
    for i in range(sm.shape[0]):
        if abs(sm[i][i]) < 0.0001:
            print("WARNING: singularity in sm:", i)
            sm[i][i] = 1e100

    return sm


#helpers:
def didx_to_nidx_dir(didx, _dof):

    return int(didx/_dof), didx%_dof


def ntag_dir_to_didx(ntag, dir, _dof, ntag2nidx):

    return ntag2nidx[ntag]*_dof + dir


def value_to_table(didxs, values, nidx2ntag, _dof):
    tup = [(didx_to_nidx_dir(didx, _dof), d) for didx, d in zip(didxs, values)]
    for key, val in groupby(sorted(tup, key=lambda x: x[0][0]), key=lambda x: x[0][0]):
        val = list(val)
        lst = [0.0]*_dof
        for (k, dir), v in val:
            lst[dir] = v
        k = nidx2ntag[val[0][0][0]]
        
        yield [k] + lst


def get_max(lst):
    max_tag = [""]*12
    max_val = [0.0]*12
    tag2row = {}
    for row in lst:
        if len(row)==2:
            vs = row[1]
        else:
            vs = row[1:12]
        tag2row[row[0]] = vs
        for i, v in enumerate(list(vs)[:12]):
            if abs(v) > max_val[i]:
                max_val[i] = v
                max_tag[i] = row[0]
    #print(set(max_tag) - set([""]), tag2row.keys(), max_val)
    return [(tag, tag2row[tag]) for tag in sorted(list(set(max_tag)-set([""])), key=lambda x: x[0])]


def to_string(lst, job='frame'):
    limit = 13
    if job == 'truss':
        limit = 4
    mim_val, max_val = 0.0, 0.0
    values = []
    for row in lst:
        if len(row)==2:
            vs = row[1]
        else:
            vs = row[1:limit]
        vals = []
        for v in vs:
            v = float(v)
            a = abs(v)
            if a > max_val:
                max_val = a
            vals += [v]
        values += [(row[0], vals)]
    base = 1.0
    base_text = ""
    if max_val > 1000:
        base = 1000
        base_text = " (x 10^3)"
    elif max_val < 0.1:
        base = 0.001
        base_text = " (x 10^-3)"
    limit = len(values[0][1])
    tag = 'node'
    if limit > 6:
        tag = 'member'
    ret_str = '%-8s\t'%tag + ' '.join(['%10d'%(i+1) for i in range(limit)]) + '\n'
    if job == 'truss' and len(values[0][1])>3:
        tag = 'member'
        ret_str = '%-8s\t'%tag + ' '.join(['%10d'%(i+1) for i in range(2)]) + '\n'
        values = [(row0, [val[0], val[3]]) for row0, val in values]
    for row0, vals in values:
        ret_str += '%-8s'%row0 + '\t' + ' '.join(['%10.3f'%(v/base) for v in vals]) + '\n'

    return base_text + '\n' + ret_str  


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