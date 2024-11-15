import itertools
import math
import time

import gurobipy as gp
import numpy as np
import pyomo.contrib.alternative_solutions as aos
import pyomo.environ as pyo
from gurobipy import GRB
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from tqdm.auto import tqdm


class MIPMol:
    """A class for generating molecules using mixed-integer programming.

    Attributes
    ----------
    atoms : list[str]
        Defines the atom-types a solution molecule can have.
    N_atoms : int
        The number of atoms in a solution molecule.
    covalences : list[int]
        The default valence number of each atom-type.
    idx_atoms : dict[int, int]
        The index of each atom-type in `atoms`.

    """

    atoms: list[str]

    @property
    def N_types(self) -> int:
        """The number of atom-types."""
        return len(self.atoms)

    def __init__(self, atoms, N_atoms, language="Gurobi"):
        # only Gurobi and Pyomo are supported
        if language not in ["Gurobi", "Pyomo"]:
            raise ValueError(f"Modeling language {language} is not supported.")
        self.language = language

        # types of atoms
        self.atoms = atoms
        # number of atoms
        self.N_atoms = N_atoms
        # get the covalence of each type of atom
        self.covalences = []
        for atom in self.atoms:
            self.covalences.append(
                Chem.rdchem.PeriodicTable.GetDefaultValence(
                    Chem.rdchem.GetPeriodicTable(), Chem.Atom(atom).GetAtomicNum()
                )
            )
        # the index of each type of atom in self.atoms
        self.idx_atoms = {}
        for idx, atom in enumerate(self.atoms):
            self.idx_atoms.setdefault(Chem.Atom(atom).GetAtomicNum(), idx)

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

        # define model and variables
        self.initialize_model()

        # add structural constraints
        self.structural_feasibility()

        # If one wants to exclude a large substructure,
        # instead of adding too many constraints,
        # put it into this list and check it in validation stage
        self.check_later = []

    # initialize model with dummy objective and variables for features
    def initialize_model(self):
        # create model and set objective as 0
        if self.language == "Gurobi":
            self.m = gp.Model()
            self.m.setObjective(expr=0)
        else:
            self.m = pyo.ConcreteModel()
            self.m.Obj = pyo.Objective(expr=0)
            self.m.Con = pyo.ConstraintList()

        # define variables for atom and bond features
        N, F = self.N_atoms, self.N_features
        self.add_variable([N, F], "X")
        self.add_variable([N, N], "A")
        self.add_variable([N, N], "DB")
        self.add_variable([N, N], "TB")

    # add variable given shape and name
    def add_variable(self, shape, name):
        assert len(shape) in [1, 2]
        if self.language == "Gurobi":
            if len(shape) == 1:
                setattr(
                    self, name, self.m.addVars(shape[0], vtype=GRB.BINARY, name=name)
                )
            else:
                setattr(
                    self,
                    name,
                    self.m.addVars(shape[0], shape[1], vtype=GRB.BINARY, name=name),
                )
        else:
            if len(shape) == 1:
                setattr(self.m, name, pyo.Var(range(shape[0]), within=pyo.Binary))
            else:
                setattr(
                    self.m,
                    name,
                    pyo.Var(range(shape[0]), range(shape[1]), within=pyo.Binary),
                )
            setattr(self, name, getattr(self.m, name))

    # add constraint given the expression and sense (<= or ==)
    # for gurobi, we also include the name of this constraint for later use if the model is infeasible
    def add_constraint(self, expr, sense, name):
        if self.language == "Gurobi":
            if sense == "==":
                self.m.addConstr(expr == 0, name=name)
            elif sense == "<=":
                self.m.addConstr(expr <= 0, name=name)
        else:
            if sense == "==":
                self.m.Con.add(expr == 0)
            elif sense == "<=":
                self.m.Con.add(expr <= 0)

    # basic constraints for structural feasibility
    def structural_feasibility(self):
        name = "structural feasibility"

        # (C1): assume that each atom exists
        for v in range(self.N_atoms):
            self.add_constraint(self.A[v, v] - 1, "==", name)

        # (C2): A is symmetric
        for u in range(self.N_atoms):
            for v in range(u + 1, self.N_atoms):
                self.add_constraint(self.A[u, v] - self.A[v, u], "==", name)

        # (C3): force connectivity of subgraphs induced by {0,1,...,v}
        for v in range(1, self.N_atoms):
            expr = self.A[v, v]
            for u in range(v):
                expr -= self.A[u, v]
            self.add_constraint(expr, "<=", name)

        # (C4): no self double bond
        for v in range(self.N_atoms):
            self.add_constraint(self.DB[v, v], "==", name)

        # (C5): DB is symmetric
        for u in range(self.N_atoms):
            for v in range(u + 1, self.N_atoms):
                self.add_constraint(self.DB[u, v] - self.DB[v, u], "==", name)

        # (C6): no self triple bond
        for v in range(self.N_atoms):
            self.add_constraint(self.TB[v, v], "==", name)

        # (C7): TB is symmetric
        for u in range(self.N_atoms):
            for v in range(u + 1, self.N_atoms):
                self.add_constraint(self.TB[u, v] - self.TB[v, u], "==", name)

        # (C8): a double/triple bond between u and v exists only when edge u-v exist
        for u in range(self.N_atoms):
            for v in range(u + 1, self.N_atoms):
                self.add_constraint(
                    self.DB[u, v] + self.TB[u, v] - self.A[u, v], "<=", name
                )

        # (C9): force one and only one type to each atom
        for v in range(self.N_atoms):
            expr = self.A[v, v]
            for f in self.idx_types:
                expr -= self.X[v, f]
            self.add_constraint(expr, "==", name)

        # (C10): force one and only one possible number of neighbors for each atom
        for v in range(self.N_atoms):
            expr = self.A[v, v]
            for f in self.idx_neighbors:
                expr -= self.X[v, f]
            self.add_constraint(expr, "==", name)

        # (C11): force one and only one possible number of hydrogens associated with each atom
        for v in range(self.N_atoms):
            expr = self.A[v, v]
            for f in self.idx_hydrogens:
                expr -= self.X[v, f]
            self.add_constraint(expr, "==", name)

        # (C12): number of neighbors calculated from A or from X should match
        for v in range(self.N_atoms):
            expr = 0.0
            for u in range(self.N_atoms):
                if u != v:
                    expr += self.A[u, v]
            for i in range(self.N_neighbors):
                expr -= (i + 1) * self.X[v, self.idx_neighbors[i]]
            self.add_constraint(expr, "==", name)

        # (C13): a double bond between u and v exists when u and v are both associated with double bond and edge u-v exists
        for u in range(self.N_atoms):
            for v in range(u + 1, self.N_atoms):
                expr = (
                    3.0 * self.DB[u, v]
                    - self.X[u, self.idx_double_bond]
                    - self.X[v, self.idx_double_bond]
                    - self.A[u, v]
                )
                self.add_constraint(expr, "<=", name)

        # (C14): a triple bond between u and v exists when u and v are both associated with triple bond and edge u-v exists
        for u in range(self.N_atoms):
            for v in range(u + 1, self.N_atoms):
                expr = (
                    3.0 * self.TB[u, v]
                    - self.X[u, self.idx_triple_bond]
                    - self.X[v, self.idx_triple_bond]
                    - self.A[u, v]
                )
                self.add_constraint(expr, "<=", name)

        # (C15): maximal number of double bonds linked to each atom
        for v in range(self.N_atoms):
            expr = 0.0
            for u in range(self.N_atoms):
                if u != v:
                    expr += self.DB[u, v]
            for i in range(self.N_types):
                expr -= (self.covalences[i] // 2) * self.X[v, self.idx_types[i]]
            self.add_constraint(expr, "<=", name)

        # (C16): double bond feature for atom v is activated when there is at least one double bond between v and another atom
        for v in range(self.N_atoms):
            expr = self.X[v, self.idx_double_bond]
            for u in range(self.N_atoms):
                if u != v:
                    expr -= self.DB[u, v]
            self.add_constraint(expr, "<=", name)

        # (C17): maximal number of triple bonds linked to each atom
        for v in range(self.N_atoms):
            expr = 0.0
            for u in range(self.N_atoms):
                if u != v:
                    expr += self.TB[u, v]
            for i in range(self.N_types):
                expr -= (self.covalences[i] // 3) * self.X[v, self.idx_types[i]]
            self.add_constraint(expr, "<=", name)

        # (C18): triple bond feature for atom v is activated when there is at least one triple bond between v and another atom
        for v in range(self.N_atoms):
            expr = self.X[v, self.idx_triple_bond]
            for u in range(self.N_atoms):
                if u != v:
                    expr -= self.TB[u, v]
            self.add_constraint(expr, "<=", name)

        # (C19): covalence equation
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
            self.add_constraint(expr, "==", name)

    # (C20): set bounds for each type of atom
    def bounds_atoms(self, lb, ub):
        for i in range(self.N_types):
            expr = 0.0
            for v in range(self.N_atoms):
                expr += self.X[v, self.idx_types[i]]
            if lb[i] is not None:
                self.add_constraint(
                    lb[i] - expr, "<=", name=f"lower bound of {self.atoms[i]}"
                )
            if ub[i] is not None:
                self.add_constraint(
                    expr - ub[i], "<=", name=f"upper bound of {self.atoms[i]}"
                )

    # (C21): set bounds for number of double bonds
    def bounds_double_bonds(self, lb_db=None, ub_db=None):
        expr = 0.0
        for u in range(self.N_atoms):
            for v in range(u + 1, self.N_atoms):
                expr += self.DB[u, v]
        if lb_db is not None:
            self.add_constraint(lb_db - expr, "<=", name="lower bound of double bonds")
        if ub_db is not None:
            self.add_constraint(expr - ub_db, "<=", name="upper bound of double bonds")

    # (C22): set bounds for number of triple bonds
    def bounds_triple_bonds(self, lb_tb=None, ub_tb=None):
        expr = 0.0
        for u in range(self.N_atoms):
            for v in range(u + 1, self.N_atoms):
                expr += self.TB[u, v]
        if lb_tb is not None:
            self.add_constraint(lb_tb - expr, "<=", name="lower bound of triple bonds")
        if ub_tb is not None:
            self.add_constraint(expr - ub_tb, "<=", name="upper bound of triple bonds")

    # (C23): set bounds for number of rings
    def bounds_rings(self, lb_ring=None, ub_ring=None):
        expr = -(self.N_atoms - 1)
        for u in range(self.N_atoms):
            for v in range(u + 1, self.N_atoms):
                expr += self.A[u, v]
        if lb_ring is not None:
            self.add_constraint(lb_ring - expr, "<=", name="lower bound of rings")
        if ub_ring is not None:
            self.add_constraint(expr - ub_ring, "<=", name="upper bound of rings")

    # extract atom/bond/(explicit)hydrogen/degree information for a SMARTS string
    def substructure_parser(self, substructure):
        atom_list = []
        bond_list = []
        hydrogen_list = []
        degree_list = []
        mol = Chem.MolFromSmarts(substructure)
        for atom in mol.GetAtoms():
            if atom.GetIsAromatic():
                raise ValueError(
                    f"Aromaticity is not supported. Please kekulize substructure {substructure} first if you still want to use Limeade."
                )
            atom_list.append([])
            hydrogen_list.append(None)
            degree_list.append(None)
            queries = atom.DescribeQuery().split("\n")
            for query in queries:
                words = query.split(" ")
                if "AtomType" in words:
                    temp_atom = int(words[words.index("AtomType") + 1])
                    atom_list[-1].append(self.idx_atoms[temp_atom])
                elif "AtomHCount" in words:
                    temp_hydrogen = int(words[words.index("AtomHCount") + 1])
                    hydrogen_list[-1] = temp_hydrogen
                elif "AtomTotalDegree" in words:
                    temp_degree = int(words[words.index("AtomTotalDegree") + 1])
                    degree_list[-1] = temp_degree
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
        return atom_list, bond_list, hydrogen_list, degree_list

    # exclude given substructures
    def exclude_substructures(self, substructures):
        for substructure in substructures:
            atom_list, bond_list, hydrogen_list, degree_list = self.substructure_parser(
                substructure
            )
            n = len(atom_list)
            if n * math.log10(self.N_atoms) > 5:
                # If one wants to exclude a large substructure,
                # instead of adding too many constraints,
                # put it into this list and check it in validation stage
                self.check_later.append(substructure)
                continue
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
                        if degree_list[i] is not None:
                            neighbor = degree_list[i] - hydrogen_list[i]
                            expr += self.X[l[i], self.idx_neighbors[neighbor - 1]]
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
                self.add_constraint(expr, "<=", name=f"exclude {substructure}")

    # include given substructures
    def include_substructures(self, substructures):
        for substructure in substructures:
            atom_list, bond_list, hydrogen_list, degree_list = self.substructure_parser(
                substructure
            )
            n = len(atom_list)

            A = {}
            for bond in bond_list:
                u, v = bond[0], bond[1]
                if u > v:
                    u, v = v, u
                A[(u, v)] = True
            self.add_variable([self.N_atoms - n], f"Y[{substructure}]")
            Y = getattr(self, f"Y[{substructure}]")
            for k in range(self.N_atoms - n):

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
                        if degree_list[i] is not None:
                            neighbor = degree_list[i] - hydrogen_list[i]
                            expr += self.X[i + k, self.idx_neighbors[neighbor - 1]]
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
                for u in range(n):
                    for v in range(u + 1, n):
                        if (u, v) not in A:
                            expr += 1 - self.A[u + k, v + k]
                            M += 1
                expr -= Y[k] * M
                self.add_constraint(-expr, "<=", name=f"include {substructure}")
            expr = -1.0
            for i in range(self.N_atoms - n):
                expr += Y[i]
            self.add_constraint(-expr, "<=", name=f"include {substructure}")

    # validation stage, remove:
    # (i) duplicated molecules, and
    # (ii) molecules with substructures in self.check_later
    def validate(self, mols):
        valid_mols = []
        uni_smiles = {}
        for mol in mols:
            mol.UpdatePropertyCache()
            smiles = Chem.MolToSmiles(mol)
            # check symmetry
            if smiles in uni_smiles:
                continue
            uni_smiles[smiles] = True
            valid = True
            # check substructures that are not represented as constraints
            for substructure in self.check_later:
                pattern = Chem.MolFromSmarts(substructure)
                if mol.HasSubstructMatch(pattern):
                    valid = False
                    break
            if valid:
                valid_mols.append(mol)
        return valid_mols

    # generate solutions within time limit for each batch using Gurobi
    def solve(self, NumSolutions, BatchSize=100, TimeLimit=600):
        if self.language != "Gurobi":
            raise ValueError(
                "Please use self.solve_pyomo when the modeling language is Pyomo."
            )
        tic = time.time()
        mols = []
        # number of batches needed
        Batch = NumSolutions // BatchSize + (NumSolutions % BatchSize != 0)
        self.m.update()
        for batch in tqdm(range(Batch)):
            # number of solutions needed for this batch
            PoolSolutions = (
                BatchSize
                if batch < Batch - 1
                else NumSolutions - (Batch - 1) * BatchSize
            )
            # set hyperparameters
            self.m.reset()
            self.m.resetParams()
            self.m.Params.OutputFlag = False
            self.m.Params.PoolSearchMode = 2
            self.m.Params.PoolSolutions = PoolSolutions
            self.m.Params.TimeLimit = TimeLimit
            self.m.Params.Seed = batch
            # solve the model
            self.m.optimize()

            # infeasibility information
            if self.m.Status == GRB.INFEASIBLE:
                if len(self.m.getConstrs()) <= 1e5:
                    self.m.computeIIS()
                    infeasible = {}
                    for c in self.m.getConstrs():
                        if c.IISConstr:
                            infeasible[c.constrname] = 1
                    print("Infeasible model. Please check the following constraints:")
                    for key in infeasible.keys():
                        if key != "structural feasibility":
                            print("    --", key)
                else:
                    print("Infeasible model.")
                return []

            # construct molecules from solutions
            for idx in range(self.m.SolCount):
                self.m.Params.SolutionNumber = idx
                mol = AllChem.EditableMol(Chem.MolFromSmiles(""))
                for v in range(self.N_atoms):
                    for f in range(self.N_features):
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
        toc = time.time()
        valid_mols = self.validate(mols)
        print(f"{len(mols)} molecules are generated after {round(toc-tic, 2)} seconds.")
        print(
            f"There are {len(valid_mols)} molecules left after removing symmetric and invalid molecules."
        )
        return valid_mols

    # generate solutions within time limit for each batch using Pyomo specified a solver
    def solve_pyomo(
        self, NumSolutions, BatchSize=100, solver="cplex_direct", solver_options={}
    ):
        if self.language != "Pyomo":
            raise ValueError(
                "Please use self.solve when the modeling language is Gurobi."
            )
        tic = time.time()
        mols = []
        # number of batches needed
        Batch = NumSolutions // BatchSize + (NumSolutions % BatchSize != 0)
        for batch in tqdm(range(Batch)):
            # number of solutions needed for this batch
            PoolSolutions = (
                BatchSize
                if batch < Batch - 1
                else NumSolutions - (Batch - 1) * BatchSize
            )
            # solve the model
            m = self.m.create_instance()
            try:
                sols = aos.enumerate_binary_solutions(
                    m,
                    num_solutions=PoolSolutions,
                    search_mode="random",
                    solver=solver,
                    solver_options=solver_options,
                    seed=batch,
                )
            except:
                print("Infeasible model.")
                return []
            # construct molecules from solutions
            for sol in sols:
                mol = AllChem.EditableMol(Chem.MolFromSmiles(""))
                for v in range(self.N_atoms):
                    for f in range(self.N_features):
                        if np.rint(sol.get_variable_name_values()[f"X[{v},{f}]"]):
                            mol.AddAtom(Chem.Atom(self.atoms[f]))
                            break
                for u in range(self.N_atoms):
                    for v in range(u + 1, self.N_atoms):
                        if np.rint(sol.get_variable_name_values()[f"A[{u},{v}]"]):
                            if np.rint(sol.get_variable_name_values()[f"DB[{u},{v}]"]):
                                mol.AddBond(u, v, Chem.BondType.DOUBLE)
                            elif np.rint(
                                sol.get_variable_name_values()[f"TB[{u},{v}]"]
                            ):
                                mol.AddBond(u, v, Chem.BondType.TRIPLE)
                            else:
                                mol.AddBond(u, v, Chem.BondType.SINGLE)
                mols.append(mol.GetMol())
        toc = time.time()
        valid_mols = self.validate(mols)
        print(f"{len(mols)} molecules are generated after {round(toc-tic, 2)} seconds.")
        print(
            f"There are {len(valid_mols)} molecules left after removing symmetric and invalid molecules."
        )
        return valid_mols
