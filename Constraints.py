import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math
import itertools
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import time
from tqdm import tqdm

class MIPMol:
    def __init__(self, atoms, N_atoms):
        self.atoms = atoms
        # get the covalence of each type of atom
        self.covalences = []
        for atom in self.atoms:
            self.covalences.append(
                Chem.rdchem.PeriodicTable.GetDefaultValence(
                    Chem.rdchem.GetPeriodicTable(), Chem.Atom(atom).GetAtomicNum()
                )
            )
        # give the index of each type of atom in self.atoms
        self.idx_atoms = {}
        for idx, atom in enumerate(self.atoms):
            self.idx_atoms.setdefault(Chem.Atom(atom).GetAtomicNum(), idx)
        # number of types of atoms, indexed from 0 to len(atoms) - 1
        self.N_types = len(self.atoms)
        self.idx_types = range(0, self.N_types)
        # number of neighbors for each atom, ranging from 1 to max(covalences)
        self.N_neighbors = max(self.covalences)
        self.idx_neighbors = range(self.N_types, self.N_types + self.N_neighbors)
        # number of hydrogens associated with each atom, ranging from 0 to max(covalences)
        self.N_hydrogens = max(self.covalences) + 1
        self.idx_hydrogens = range(
            self.N_types + self.N_neighbors,
            self.N_types + self.N_neighbors + self.N_hydrogens,
        )
        # number of features, including two features representing double bond and triple bond
        self.N_features = self.N_types + self.N_neighbors + self.N_hydrogens + 2
        # index of double bond feature
        self.idx_double_bond = self.N_features - 2
        # index of triple bond feature
        self.idx_triple_bond = self.N_features - 1

        # print(self.covalences)
        # print(self.idx_atoms)
        # print(self.N_types, self.N_neighbors, self.N_hydrogens)
        # print(self.N_features)
        # print(self.idx_types, self.idx_neighbors, self.idx_hydrogens, self.idx_double_bond, self.idx_triple_bond)

        self.N_atoms = N_atoms
        self.m = gp.Model("MIP_CAMD")
        self.X = self.m.addVars(
            range(self.N_atoms), range(self.N_features), vtype=GRB.BINARY, name="X"
        )
        self.A = self.m.addVars(
            range(self.N_atoms), range(self.N_atoms), vtype=GRB.BINARY, name="A"
        )
        self.DB = self.m.addVars(
            range(self.N_atoms), range(self.N_atoms), vtype=GRB.BINARY, name="DB"
        )
        self.TB = self.m.addVars(
            range(self.N_atoms), range(self.N_atoms), vtype=GRB.BINARY, name="TB"
        )

        self.structural_feasibility()

    # basic constraints for structural feasibility
    def structural_feasibility(self):
        name = "structural feasibility"
        # m.addConstr(A[0,0]==1)
        # m.addConstr(A[1,1]==1)
        # m.addConstr((m.A[v,v] >= m.A[v+1,v+1] for v in range(N_atoms-1)))

        # assume that each atom exists
        for v in range(self.N_atoms):
            self.m.addConstr(self.A[v, v] == 1, name=name)

        # A is symmetric
        for u in range(self.N_atoms):
            for v in range(u + 1, self.N_atoms):
                self.m.addConstr(self.A[u, v] == self.A[v, u], name=name)

        # for v in range(self.N_atoms):
        #    expr = (self.N_atoms - 1) * self.A[v,v]
        #    for u in range(self.N_atoms):
        #        if u != v:
        #            expr -= self.A[u,v]
        #    self.m.addConstr(expr >= 0)

        # force connectivity of subgraphs induced by {0,1,...,v}
        for v in range(1, self.N_atoms):
            expr = self.A[v, v]
            for u in range(v):
                expr -= self.A[u, v]
            self.m.addConstr(expr <= 0, name=name)

        # no self double bond
        for v in range(self.N_atoms):
            self.m.addConstr((self.DB[v, v] == 0), name=name)

        # DB is symmetric
        for u in range(self.N_atoms):
            for v in range(u + 1, self.N_atoms):
                self.m.addConstr((self.DB[u, v] == self.DB[v, u]), name=name)

        # no self triple bond
        for v in range(self.N_atoms):
            self.m.addConstr((self.TB[v, v] == 0), name=name)

        # TB is symmetric
        for u in range(self.N_atoms):
            for v in range(u + 1, self.N_atoms):
                self.m.addConstr((self.TB[u, v] == self.TB[v, u]), name=name)

        # a double/triple bond between u and v exists only when edge u-v exist
        for u in range(self.N_atoms):
            for v in range(u + 1, self.N_atoms):
                self.m.addConstr((self.DB[u, v] + self.TB[u, v] <= self.A[u, v]), name=name)

        # force one and only one type to each atom
        for v in range(self.N_atoms):
            expr = self.A[v, v]
            for f in self.idx_types:
                expr -= self.X[v, f]
            self.m.addConstr(expr == 0, name=name)

        # force one and only one possible number of neighbors for each atom
        for v in range(self.N_atoms):
            expr = self.A[v, v]
            for f in self.idx_neighbors:
                expr -= self.X[v, f]
            self.m.addConstr(expr == 0, name=name)

        # force one and only one possible number of hydrogens associated with each atom
        for v in range(self.N_atoms):
            expr = self.A[v, v]
            for f in self.idx_hydrogens:
                expr -= self.X[v, f]
            self.m.addConstr(expr == 0, name=name)

        # number of neighbors calculated from A or from X should match
        for v in range(self.N_atoms):
            expr = 0.0
            for u in range(self.N_atoms):
                if u != v:
                    expr += self.A[u, v]
            for i in range(self.N_neighbors):
                expr -= (i + 1) * self.X[v, self.idx_neighbors[i]]
            self.m.addConstr(expr == 0, name=name)

        # a double bond between u and v exists when u and v are both associated with double bond and edge u-v exists
        for u in range(self.N_atoms):
            for v in range(u + 1, self.N_atoms):
                self.m.addConstr(
                    (
                        3.0 * self.DB[u, v]
                        - self.X[u, self.idx_double_bond]
                        - self.X[v, self.idx_double_bond]
                        - self.A[u, v]
                        <= 0
                    ), name=name
                )

        # a triple bond between u and v exists when u and v are both associated with triple bond and edge u-v exists
        for u in range(self.N_atoms):
            for v in range(u + 1, self.N_atoms):
                self.m.addConstr(
                    (
                        3.0 * self.TB[u, v]
                        - self.X[u, self.idx_triple_bond]
                        - self.X[v, self.idx_triple_bond]
                        - self.A[u, v]
                        <= 0
                    ), name=name
                )

        # maximal number of double bonds linked to each atom
        for v in range(self.N_atoms):
            expr = 0.0
            for u in range(self.N_atoms):
                if u != v:
                    expr += self.DB[u, v]
            for i in range(self.N_types):
                expr -= (self.covalences[i] // 2) * self.X[v, self.idx_types[i]]
            self.m.addConstr(expr <= 0, name=name)

        # double bond feature for atom v is activated when there is at least one double bond between v and another atom
        for v in range(self.N_atoms):
            expr = self.X[v, self.idx_double_bond]
            for u in range(self.N_atoms):
                if u != v:
                    expr -= self.DB[u, v]
            self.m.addConstr(expr <= 0, name=name)

        # maximal number of triple bonds linked to each atom
        for v in range(self.N_atoms):
            expr = 0.0
            for u in range(self.N_atoms):
                if u != v:
                    expr += self.TB[u, v]
            for i in range(self.N_types):
                expr -= (self.covalences[i] // 3) * self.X[v, self.idx_types[i]]
            self.m.addConstr(expr <= 0, name=name)

        # triple bond feature for atom v is activated when there is at least one triple bond between v and another atom
        for v in range(self.N_atoms):
            expr = self.X[v, self.idx_triple_bond]
            for u in range(self.N_atoms):
                if u != v:
                    expr -= self.TB[u, v]
            self.m.addConstr(expr <= 0, name=name)

        # covalence equation
        for v in range(self.N_atoms):
            expr = 0.0
            for i in range(self.N_types):
                expr += self.covalences[i] * self.X[v, self.idx_types[i]]
            for i in range(self.N_neighbors):
                expr -= (i + 1) * self.X[v, self.idx_neighbors[i]]
            for i in range(self.N_hydrogens):
                expr -= i * self.X[v, self.idx_hydrogens[i]]
            for u in range(self.N_atoms):
                if u != v:
                    expr -= self.DB[u, v]
            for u in range(self.N_atoms):
                if u != v:
                    expr -= 2.0 * self.TB[u, v]
            self.m.addConstr(expr == 0, name=name)

    # set bounds for each type of atom
    def bounds_atoms(self, lb, ub):
        for i in range(self.N_types):
            expr = 0.0
            for v in range(self.N_atoms):
                expr += self.X[v, self.idx_types[i]]
            if lb[i] is not None:
                self.m.addConstr(expr >= lb[i], name = f"lower bound of {self.atoms[i]}")
            if ub[i] is not None:
                self.m.addConstr(expr <= ub[i], name = f"upper bound of {self.atoms[i]}")

    # set bounds for number of double bonds
    def bounds_double_bonds(self, lb_db=None, ub_db=None):
        expr = 0.0
        for u in range(self.N_atoms):
            for v in range(u + 1, self.N_atoms):
                expr += self.DB[u, v]
        if lb_db is not None:
            self.m.addConstr(expr >= lb_db, name = "lower bound of double bonds")
        if ub_db is not None:
            self.m.addConstr(expr <= ub_db, name = "upper bound of double bonds")

    # set bounds for number of triple bonds
    def bounds_triple_bonds(self, lb_tb=None, ub_tb=None):
        expr = 0.0
        for u in range(self.N_atoms):
            for v in range(u + 1, self.N_atoms):
                expr += self.TB[u, v]
        if lb_tb is not None:
            self.m.addConstr(expr >= lb_tb, name = "lower bound of triple bonds")
        if ub_tb is not None:
            self.m.addConstr(expr <= ub_tb, name = "upper bound of triple bonds")

    # set bounds for number of rings
    def bounds_rings(self, lb_ring=None, ub_ring=None):
        expr = -(self.N_atoms - 1)
        for u in range(self.N_atoms):
            for v in range(u + 1, self.N_atoms):
                expr += self.A[u, v]
        if lb_ring is not None:
            self.m.addConstr(expr >= lb_ring, name = "lower bound of rings")
        if ub_ring is not None:
            self.m.addConstr(expr <= ub_ring, name = "upper bound of rings")

    # extract atom/bond/(explicit)hydrogen information from SMILES
    def smiles_parser(self, smiles):
        atom_list = []
        bond_list = []
        hydrogen_list = []
        mol = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(mol)
        for atom in mol.GetAtoms():
            atom_list.append([self.idx_atoms[atom.GetAtomicNum()]])
            if atom.GetNumExplicitHs():
                hydrogen_list.append(atom.GetNumExplicitHs())
            else:
                hydrogen_list.append(None)
        for bond in mol.GetBonds():
            bond_list.append(
                [
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    int(bond.GetBondType()),
                ]
            )
        return atom_list, bond_list, hydrogen_list

    # extract atom/bond/(explicit)hydrogen information from SMARTS
    def smarts_parser(self, smarts):
        atom_list = []
        bond_list = []
        hydrogen_list = []
        mol = Chem.MolFromSmarts(smarts)
        # Chem.Kekulize(mol)
        for atom in mol.GetAtoms():
            atom_list.append([])
            hydrogen_list.append(None)
            queries = atom.DescribeQuery().split("\n")
            for query in queries:
                words = query.split(" ")
                if "AtomType" in words:
                    temp = int(words[words.index("AtomType") + 1])
                    atom_list[-1].append(self.idx_atoms[temp])
                elif "AtomHCount" in words:
                    temp = int(words[words.index("AtomHCount") + 1])
                    hydrogen_list[-1] = temp
            if not len(atom_list[-1]):
                atom_list[-1] = list(range(self.N_types))
        for bond in mol.GetBonds():
            bond_list.append(
                [
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    int(bond.GetBondType()),
                ]
            )
        return atom_list, bond_list, hydrogen_list

    # exclude given substructures (list of SMILES/SMARTS)
    def exclude_substructures(self, substructures, language="SMILES"):
        for substructure in substructures:
            if language == "SMILES":
                atom_list, bond_list, hydrogen_list = self.smiles_parser(substructure)
            elif language == "SMARTS":
                atom_list, bond_list, hydrogen_list = self.smarts_parser(substructure)
            else:
                raise ValueError("Invalid language for describing molecular patterns.")
            n = len(atom_list)
            if math.comb(self.N_atoms, n) > 1e5:
                raise ValueError("Too many constraints.")

            idx_list = [range(self.N_atoms) for _ in range(n)]
            for l in itertools.product(*idx_list):
                if len(set(l)) < n:
                    continue
                # big-M coefficient
                M = 0
                expr = 0
                for i in range(n):
                    for v in atom_list[i]:
                        expr += self.X[l[i], v]
                    M += 1
                    if hydrogen_list[i] is not None:
                        expr += self.X[l[i], self.idx_hydrogens[hydrogen_list[i]]]
                        M += 1
                for bond in bond_list:
                    u, v, bond_type = bond[0], bond[1], bond[2]
                    if bond_type == 0:
                        expr += self.A[l[u], l[v]]
                    elif bond_type == 1:
                        expr += (
                            self.A[l[u], l[v]]
                            - self.DB[l[u], l[v]]
                            - self.TB[l[u], l[v]]
                        )
                    elif bond_type == 2:
                        expr += self.DB[l[u], l[v]]
                    elif bond_type == 3:
                        expr += self.TB[l[u], l[v]]
                    else:
                        raise ValueError("Invalid bond type.")
                    M += 1
                expr -= M - 1
                self.m.addConstr(expr <= 0, name=f"exclude {substructure}")

    # include given substructures (list of SMILES\SMARTS)
    def include_substructures(self, substructures, language="SMILES"):
        for substructure in substructures:
            if language == "SMILES":
                atom_list, bond_list, hydrogen_list = self.smiles_parser(substructure)
            elif language == "SMARTS":
                atom_list, bond_list, hydrogen_list = self.smarts_parser(substructure)
            else:
                raise ValueError("Invalid language for describing molecular patterns.")

            n = len(atom_list)
            Y = []
            idx_Y = 0
            for k in range(self.N_atoms - n):
                Y.append(
                    self.m.addVar(
                        vtype=GRB.BINARY, name="Y[%s,%d]" % (substructure, idx_Y)
                    )
                )
                idx_Y += 1
                # big-M coefficient
                M = 0
                expr = 0
                for i in range(n):
                    for v in atom_list[i]:
                        expr += self.X[i + k, v]
                    M += 1
                    if hydrogen_list[i]:
                        expr += self.X[i + k, self.idx_hydrogens[hydrogen_list[i]]]
                        M += 1
                for bond in bond_list:
                    u, v, bond_type = bond[0] + k, bond[1] + k, bond[2]
                    # any bond
                    if bond_type == 0:
                        expr += self.A[u, v]
                    # single bond
                    elif bond_type == 1:
                        expr += self.A[u, v] - self.DB[u, v] - self.TB[u, v]
                    # double bond
                    elif bond_type == 2:
                        expr += self.DB[u, v]
                    # triple bond
                    elif bond_type == 3:
                        expr += self.TB[u, v]
                    # invalid bond
                    else:
                        raise ValueError("Invalid bond type.")
                    M += 1
                expr -= Y[-1] * M
                self.m.addConstr(expr >= 0, name=f"include {substructure}")
            expr = -1.0
            for i in range(idx_Y):
                expr += Y[i]
            self.m.addConstr(expr >= 0, name=f"include {substructure}")

    # generate solutions within timelimit
    def solve(self, NumSolutions=100, TimeLimit=3600, BatchSize=100):
        mols = []
        tic = time.time()
        batchs = NumSolutions // BatchSize
        temp = 0
        for batch in tqdm(range(batchs)):
            self.m.reset()
            self.m.resetParams()
            self.m.Params.OutputFlag = False
            self.m.Params.Seed = batch
            self.m.Params.PoolSearchMode = 2
            self.m.Params.PoolSolutions = BatchSize
            self.m.Params.SolutionNumber = BatchSize
            self.m.Params.TimeLimit = TimeLimit
            self.m.optimize()
            if self.m.Status == GRB.INFEASIBLE:
                if len(self.m.getConstrs()) <= 100000:
                    self.m.computeIIS()
                    infeasible = {}
                    for c in self.m.getConstrs():
                        if c.IISConstr: 
                            infeasible[c.constrname] = 1
                    print('Infeasible model. Please check the following constraints:')
                    for key in infeasible.keys():
                        if key != "structural feasibility":
                            print("    --", key)
                else:
                    print("Infeasible model.")
                return []
            tic1 = time.time()
            N, F = self.N_atoms, self.N_features
            for idx in range(self.m.Params.PoolSolutions):
                self.m.Params.SolutionNumber = idx
                mol = AllChem.EditableMol(Chem.MolFromSmiles(""))
                for v in range(N):
                    for f in range(F):
                        if np.rint(self.X[v, f].Xn):
                            mol.AddAtom(Chem.Atom(self.atoms[f]))
                            break
                for u in range(self.N_atoms):
                    for v in range(u + 1, self.N_atoms):
                        if np.rint(self.A[u, v].Xn):
                            if np.rint(self.DB[u, v].Xn):
                                mol.AddBond(u, v, Chem.BondType.DOUBLE)
                            elif np.rint(self.TB[u, v].Xn):
                                mol.AddBond(u, v, Chem.BondType.TRIPLE)
                            else:
                                mol.AddBond(u, v, Chem.BondType.SINGLE)      
                mols.append(mol.GetMol())
            toc1 = time.time()
            temp += toc1 - tic1
        toc = time.time()
        return mols, toc - tic, temp
