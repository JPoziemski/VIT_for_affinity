import MDAnalysis as mda
import os
import numpy as np
import argparse
import rdkit.Chem
from functools import partial
from rdkit import Chem
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from MDAnalysis.topology.guessers import guess_types, guess_bonds
from MDAnalysis.analysis.distances import distance_array
from utils import run_in_parallel
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def check_if_ligand(universe):
    if len(universe.residues.resnames) == 1:
        return True
    else:
        return False

class AtomType:
    PROPS = ["CA", "C", "Cback", "Nback", "N", "Oback", "O", "S", 'Halogen']
    backbone_atoms_convertion = {'N': "Nback",
                                 'O': 'Oback',
                                 'C': 'Cback',
                                 'CA': 'CA'}

    def __init__(self, universe: mda.Universe, props=("CA", "C", "Cback", "Nback", "N", "Oback", "O", "S", 'Halogen')):
        self.universe = universe
        self.props = props
        self.__encoded_atoms = np.zeros((max(self.universe.atoms.ids), len(props)))

    def __get_backbone(self):
        return self.universe.select_atoms("protein and backbone")

    def process_backbone(self,):
        backbone = self.__get_backbone()
        for atom in backbone:
            converted_name = self.backbone_atoms_convertion[atom.name]
            prop_id = self.props.index(converted_name)
            encoded_atom_type = np.zeros(len(self.props))
            encoded_atom_type[prop_id] = 1
            self.__encoded_atoms[atom.id-1] = encoded_atom_type

    def process_remaining_atoms(self):
        elements_to_search = ["O", "C", "S", "N"]
        for element in elements_to_search:
            matched_elements = self.universe.select_atoms(f"element {element}")
            for atom in matched_elements:
                if not self.__encoded_atoms[atom.id-1].any():
                    prop_id = self.props.index(atom.element)
                    encoded_atom_type = np.zeros(len(self.props))
                    encoded_atom_type[prop_id] = 1

                    self.__encoded_atoms[atom.id - 1] = encoded_atom_type

    def search_halogens(self):

        if not 'Halogen' in self.props:
            return
        halogen_id =  self.props.index('Halogen')
        halogen_encoded = np.zeros(len(self.props))
        halogen_encoded[halogen_id] = 1

        matched_atoms = self.universe.select_atoms('element Cl or element Br or element I or element F')
        for atom in matched_atoms:
            self.__encoded_atoms[atom.id-1] = halogen_encoded

    def encode_atoms(self):
        self.process_backbone()
        self.process_remaining_atoms()
        self.search_halogens()
        self.__encoded_atoms = self.get_heavy_atoms()


    @property
    def encoded_atoms(self):
        return self.__encoded_atoms
    def check(self):
        pass

    def get_heavy_atoms(self):
        heavy_atoms = self.universe.select_atoms("not element H")
        ids = [idx-1 for idx in heavy_atoms.atoms.ids]
        encoded_atoms = np.take(self.__encoded_atoms, ids, 0)
        return encoded_atoms

class SecondaryStructure:
    dssp_codes = np.array(["H", 'B', 'E', 'G', 'I', 'T', 'S', '-'])
    def __init__(self, universe, protein_path):
        self.universe = universe
        self.protein_path = protein_path

        self.structure_encoder = OneHotEncoder(sparse_output=False)
        self.structure_encoder.fit(self.dssp_codes.reshape(-1, 1))
        self.encoded_atoms = np.zeros((max(self.universe.atoms.ids), len(self.dssp_codes)))

    def parse_dssp_data(self):
        dssp_data = dssp_dict_from_pdb_file(self.protein_path)[0]
        aa_ids = [idx[1][1] for idx in dssp_data.keys()]
        aa_dssp_keys = np.array([data[1] for data in dssp_data.values()]).reshape(-1, 1)
        encoded_structures = self.structure_encoder.transform(aa_dssp_keys)
        structure_dict = dict(map(lambda i, j: (i, j), aa_ids, encoded_structures))
        return structure_dict

    def assign_properties_to_atoms(self):
        if check_if_ligand(self.universe):
            self.encoded_atoms = self.get_heavy_atoms()
            return
        structure_dict = self.parse_dssp_data()
        for atom in self.universe.atoms:
            prop_encoded = structure_dict.get(atom.resid, None)
            if not prop_encoded is None:
                self.encoded_atoms[atom.id-1] = prop_encoded

        self.encoded_atoms = self.encoded_atoms[~np.all(self.encoded_atoms == 0, axis=1)]

    def get_heavy_atoms(self):
        heavy_atoms = self.universe.select_atoms("not element H")
        ids = [idx-1 for idx in heavy_atoms.atoms.ids]
        
        encoded_atoms = np.take(self.encoded_atoms, ids, 0)
        return encoded_atoms

class AAType:

    def __init__(self, universe):
        self.universe = universe
        aa_groups = list(self.get_aa_groups().keys())
        self.aa_type_encoder = MultiLabelBinarizer()
        self.aa_type_encoder.fit([aa_groups])
        self.encoded_atoms = np.zeros((max(self.universe.atoms.ids), len(aa_groups)))

    def get_aa_groups(self):
        aa_groups = {
            "aromatic": ["PHE", "TRP", "TYR"],
            "charged": ["ARG", "LYS", "ASP", "GLU"],
            "hydrophobic": ["ALA", "ILE", "LEU", "MET", "PHE", "VAL", "PRO", "GLY"],
            "polar": ["GLN", "ASN", "HIS", "SER", "THR", "TYR", "CYS"]
        }
        return aa_groups

    def invert_aa_group_dict(self):
        inverted_dict = defaultdict(list)
        for key, values in self.get_aa_groups().items():
            for aa_code in values:
                inverted_dict[aa_code].append(key)
        return inverted_dict

    def assign_type_to_residue(self):
        inverted_aa_group_dict = self.invert_aa_group_dict()
        aa_types = [inverted_aa_group_dict[res] for res in self.universe.residues.resnames]
        aa_types_encoded = self.aa_type_encoder.transform(aa_types)
        aa_types_dict = dict(map(lambda i, j: (i, j), self.universe.residues.resids, aa_types_encoded))
        return aa_types_dict

    def assign_properties_to_atoms(self):
        if check_if_ligand(self.universe):
            self.encoded_atoms = self.get_heavy_atoms()
            return
        aa_types_dict = self.assign_type_to_residue()
        for atom in self.universe.atoms:
            prop_encoded = aa_types_dict.get(atom.resid, None)
            if not prop_encoded is None:
                self.encoded_atoms[atom.id - 1] = prop_encoded

        self.encoded_atoms = self.encoded_atoms[~np.all(self.encoded_atoms == 0, axis=1)]


    def get_heavy_atoms(self):
        heavy_atoms = self.universe.select_atoms("not element H")
        ids = [idx-1 for idx in heavy_atoms.atoms.ids]
        encoded_atoms = np.take(self.encoded_atoms, ids, 0)
        return encoded_atoms

def get_proper_atom_indexes_from_rdk_universe(rdk_universe):
    proper_idx = []
    for at in rdk_universe.GetAtoms():
        symbol = at.GetSymbol()

        try:
            res_name = at.GetPDBResidueInfo().GetResidueName()
            hetatm = at.GetPDBResidueInfo().GetIsHeteroAtom()
        except AttributeError:
            res_name = 'LIG'
            hetatm = False

        if symbol != 'H' and res_name != 'HOH' and len(res_name.strip()) ==3 and not hetatm:
            proper_idx.append(at.GetIdx())
    return proper_idx


class AtomProp:
    props = ["heavy_valence", "aromatic", "ring", "valence", "charge", "donor", "acceptor", "hydrophobic","hybridization"]
    ACCEPTOR_SMARTS = '[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]'
    DONOR_SMARTS = '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]'
    HYDROPHOBIC_SMARTS = '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]'

    def __init__(self, rdk_universe):
        self.rdk_universe = rdk_universe
        self.max_atom_num = len(self.rdk_universe.GetAtoms())
        self.encoded_atoms = np.zeros((self.max_atom_num, len(self.props)))
        self.heavy_atoms_indexes = get_proper_atom_indexes_from_rdk_universe(self.rdk_universe)
        self._donor_indexes = ()
        self._acceptor_indexes = ()
        self.prepare()

        self.atom_functions = {"heavy_valence": lambda atom: AtomProp.get_heavy_valence(atom),
                          "aromatic": lambda atom: int(atom.GetIsAromatic()),
                          "ring": lambda atom: int(atom.IsInRing()),
                          "valence": lambda atom: atom.GetTotalValence(),
                          "charge": lambda atom: atom.GetFormalCharge(),
                          "donor": lambda atom: int(atom.GetIdx() in self._donor_indexes),
                            "acceptor": lambda atom: int(atom.GetIdx() in self._acceptor_indexes),
                               "hydrophobic": lambda atom: int(atom.GetIdx() in self._hydrophobic_indexes),
                        "hybridization": lambda atom: int(atom.GetHybridization())}

    def prepare(self):

        if "donor" in self.props:
            donor_indexes = self.rdk_universe.GetSubstructMatches(rdkit.Chem.MolFromSmarts(self.DONOR_SMARTS))
            donor_indexes = tuple(item[0] for item in donor_indexes)
            self._donor_indexes = donor_indexes

        if "acceptor" in self.props:
            acceptor_indexes = self.rdk_universe.GetSubstructMatches(rdkit.Chem.MolFromSmarts(self.ACCEPTOR_SMARTS))
            acceptor_indexes = tuple(item[0] for item in acceptor_indexes)
            self._acceptor_indexes = acceptor_indexes

        if "hydrophobic" in self.props:
            hydrophobic_indexes = self.rdk_universe.GetSubstructMatches(rdkit.Chem.MolFromSmarts(self.HYDROPHOBIC_SMARTS))
            hydrophobic_indexes = tuple(item[0] for item in hydrophobic_indexes)
            self._hydrophobic_indexes = hydrophobic_indexes

    def encode_atoms(self):
        for atom_id in self.heavy_atoms_indexes:
            atom = self.rdk_universe.GetAtomWithIdx(atom_id)
            for prop_id, prop in enumerate(self.props):
                prop_function = self.atom_functions[prop]
                self.encoded_atoms[atom.GetIdx(), prop_id] = prop_function(atom)

        self.encoded_atoms = np.take(self.encoded_atoms, self.heavy_atoms_indexes, 0)


    @staticmethod
    def get_heavy_valence(atom):
        heavy_valence = len([at.GetSymbol() for at in atom.GetNeighbors() if at.GetSymbol() != 'H'])
        return heavy_valence


def repair_universe(universe):
    guessed_elements = guess_types(universe.atoms.names)
    universe.add_TopologyAttr('elements', guessed_elements)

    guessed_bonds = guess_bonds(universe.atoms, universe.atoms.positions)
    universe.add_TopologyAttr('bonds', guessed_bonds)
    return universe


def create_universe_for_crystal_structure(pocket_path, ligand_path):
    protein_universe = mda.Universe(pocket_path)
    protein_universe = repair_universe(protein_universe)

    ligand_universe = mda.Universe(ligand_path)
    lig_resname = ligand_universe.residues.resnames[0]

    ligand_universe = clean_universe(ligand_universe, lig_resname)
    protein_universe = clean_universe(protein_universe, lig_resname)

    return protein_universe, ligand_universe


def clean_universe(universe, lig_resname):
    repaired_universe = universe.select_atoms(f'protein or resname {lig_resname}')
    repaired_universe = repaired_universe.select_atoms(f'not element H')
    repaired_universe = repaired_universe.select_atoms(f'not type HOH')
    repaired_universe = repaired_universe.select_atoms(f'not resname MSE')
    return repaired_universe


def get_ligand_geometric_center(universe):
    ligand_cog = universe.center_of_geometry()
    return ligand_cog

def encode_universe(universe, rdk_universe, pdb_path, structure_type):
    at = AtomType(universe)
    at.encode_atoms()
    atom_rep = at.encoded_atoms

    ss = SecondaryStructure(universe, pdb_path)
    ss.assign_properties_to_atoms()
    ss_rep = ss.encoded_atoms

    aa = AAType(universe)
    aa.assign_properties_to_atoms()
    aa_rep = aa.encoded_atoms

    ap = AtomProp(rdk_universe)
    ap.encode_atoms()
    ap_rep = ap.encoded_atoms



    protein_ligand_feature = add_protein_ligand_feature( len(atom_rep), structure_type)

    feature_matrix = np.concatenate((atom_rep, ss_rep, aa_rep, ap_rep, protein_ligand_feature), axis=1)
    feature_matrix = feature_matrix[~np.all(feature_matrix == 0, axis=1)]
    return feature_matrix

def add_protein_ligand_feature( natoms, t):
    matrix_data = np.zeros((natoms, 2))

    if t == "ligand":
        matrix_data[:, 1] = 1
    else:
        matrix_data[:, 0] = 1

    return matrix_data


def create_rdkit_universe(path):
    ext = os.path.splitext(path)[1]
    if ext == ".pdb":
        rdk_universe = rdkit.Chem.MolFromPDBFile(path, sanitize=True, removeHs=False)
    else:
        rdk_universe = rdkit.Chem.MolFromMol2File(path, sanitize=False, removeHs=False)
        Chem.SanitizeMol(rdk_universe, sanitizeOps=Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION)
        if rdk_universe is None:
            path = path.replace("mol2", "sdf")
            rdk_universe = rdkit.Chem.SDMolSupplier(path, sanitize=True, removeHs=False)[0]
    return rdk_universe

def compare_atoms(mda_universe, rdk_universe):
    mda_atoms = list(mda_universe.atoms.elements)
    rdk_atoms = [at.GetSymbol() for at in rdk_universe.GetAtoms()]
    if mda_atoms == rdk_atoms:
        return True
    else:
        return False

def check_compatibility(rdk_universe, mda_universe, ligand_code):
    if rdk_universe is None:
        print("rdk unverse is none")
        return False

    rdk_universe = rdkit.Chem.RemoveHs(rdk_universe)

    proper_idx = get_proper_atom_indexes_from_rdk_universe(rdk_universe)

    rdk_cleaned_symbols = [at.GetSymbol() for at in rdk_universe.GetAtoms() if at.GetIdx() in proper_idx]
    try:
        mda_universe = clean_universe(mda_universe, ligand_code)
    except:
        return False
    mda_atoms = list(mda_universe.atoms.elements)

    cleaned_univ_com = rdk_cleaned_symbols == mda_atoms

    return cleaned_univ_com

class DistanceEncoding:
    def __init__(self, universes):
        self.protein_universe = universes[0]
        self.ligand_universe = universes[1]

    def encode_atoms(self):
        dist_arr = distance_array(self.protein_universe.positions, self.ligand_universe.positions)
        dist_arr = np.min(dist_arr, axis=1)
        dist_arr = np.pad(dist_arr, (0, len(self.ligand_universe.atoms.ids)), mode='constant')
        dist_arr = dist_arr.reshape((dist_arr.shape[0], 1))
        
        return dist_arr
        
def create_complex_representation(pocket_path, ligand_path):

    protein_universe, ligand_universe = create_universe_for_crystal_structure(pocket_path, ligand_path)
    rdk_protein, rdk_ligand = create_rdkit_universe(pocket_path), create_rdkit_universe(ligand_path)
    ligand_code = ligand_universe.residues.resnames[0]

    if not check_compatibility(rdk_protein, protein_universe, ligand_code):
        raise ValueError(f" rdk and mda protein incompatibility")


    lig_cog = get_ligand_geometric_center(ligand_universe)
    ligand_pos = ligand_universe.atoms.positions - lig_cog
    protein_pos = protein_universe.atoms.positions - lig_cog

    protein_feature_matrix = encode_universe(protein_universe, rdk_protein, pocket_path, "protein")
    ligand_feature_matrix = encode_universe(ligand_universe, rdk_ligand, pocket_path, "ligand")

    feature_matrix = np.concatenate((protein_feature_matrix, ligand_feature_matrix), axis=0)
    positions_matrix = np.concatenate((protein_pos, ligand_pos), axis=0)

    ds = DistanceEncoding((protein_universe, ligand_universe))
    dist_array = ds.encode_atoms()
    feature_matrix = np.concatenate((feature_matrix, dist_array), axis=1)
  
    return positions_matrix, feature_matrix


def get_pocket_ligand_paths(pdb, input_dir):
    protein_path = os.path.join(input_dir, pdb, f"{pdb}_protein.pdb")
    ligand_path = os.path.join(input_dir, pdb, f"{pdb}_ligand.mol2")
    return protein_path, ligand_path


def get_grid_from_pdb(pdb, input_dir, output_dir):
    protein_path, ligand_path = get_pocket_ligand_paths(pdb, input_dir)
    if os.path.exists(os.path.join(output_dir, f"{pdb}.npz")):
        return

    try:
        positions_matrix, feature_matrix = create_complex_representation(protein_path, ligand_path)
    except Exception as e:
        print(pdb, "error in processing")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.savez(os.path.join(output_dir, pdb), positions=positions_matrix, features=feature_matrix)


def get_all_pdbs_from_dir(dir_path):
    pdbs = list(p for p in os.listdir(dir_path) if len(p)==4)
    return pdbs


if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description="Train ViT model on grid dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Diectory with complexes with pdbbind directory structure")
    parser.add_argument("--output_dir", type=str, required=True, help="directory to save final structures")
    args = parser.parse_args()
    
    pdbs = get_all_pdbs_from_dir(args.input_dir)
    grid_func = partial(get_grid_from_pdb, input_dir = args.input_dir, output_dir= args.output_dir)
    run_in_parallel(grid_func, pdbs)

