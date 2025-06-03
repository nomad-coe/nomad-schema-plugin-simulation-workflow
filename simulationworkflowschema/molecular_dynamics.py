#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from typing import Any, Callable, Optional
from itertools import chain
from collections import namedtuple
import numpy as np
from array import array
from scipy import sparse
from scipy.stats import linregress
import networkx
import numpy as np
import MDAnalysis
from MDAnalysis.core.topology import Topology
from MDAnalysis.core.universe import Universe
import MDAnalysis.analysis.rdf as MDA_RDF
from MDAnalysis.core._get_readers import get_reader_for

from nomad.datamodel.data import ArchiveSection
from nomad.metainfo import (
    SubSection,
    Section,
    Quantity,
    MEnum,
    Reference,
    MSection,
)
from nomad.datamodel.hdf5 import HDF5Dataset
from nomad.datamodel.metainfo.workflow import Link
from runschema.system import System, AtomsGroup
from runschema.calculation import (
    RadiusOfGyration as RadiusOfGyrationCalculation,
    RadiusOfGyrationValues as RadiusOfGyrationValuesCalculation,
)
from nomad.utils import get_logger
from nomad.units import ureg
from nomad import atomutils
from .general import (
    SimulationWorkflowMethod,
    SimulationWorkflowResults,
    SerialSimulation,
    WORKFLOW_METHOD_NAME,
    WORKFLOW_RESULTS_NAME,
)
from .thermodynamics import ThermodynamicsResults


LOGGER = get_logger(__name__)


class BeadGroup(object):
    # see https://github.com/MDAnalysis/mdanalysis/issues/1891#issuecomment-387138110
    # by @richardjgowers with performance improvements
    def __init__(self, atoms, compound='fragments'):
        """Initialize with an AtomGroup instance.
        Will split based on keyword 'compounds' (residues or fragments).

        self._atoms: AtomGroup (total number of atoms)
        self.compound: str (dictates type of grouping)
        self._nbeads: int (total number of beads)
        self.positions: list (total number of "compounds")
        """
        self._atoms = atoms
        self.compound = compound
        self._nbeads = len(getattr(self._atoms, self.compound))
        # for caching
        self._cache = {}
        self._cache['positions'] = None
        self.__last_frame = None

    def __len__(self):
        return self._nbeads

    @property
    def positions(self):
        # cache positions for current frame
        if self.universe.trajectory.frame != self.__last_frame:
            self._cache['positions'] = (
                self._atoms.center_of_mass(unwrap=True, compound=self.compound)
                if self._nbeads != 0
                else []
            )
            self.__last_frame = self.universe.trajectory.frame
        return self._cache['positions']

    @property  # type: ignore
    @MDAnalysis.lib.util.cached('universe')
    def universe(self):
        return self._atoms.universe


def get_bond_list_from_model_contributions(
    sec_run: MSection, method_index: int = -1, model_index: int = -1
) -> list[tuple]:
    """
    Generates bond list of tuples using the list of bonded force field interactions stored under run[].method[].force_field.model[].

    bond_list: List[tuple]
    """
    contributions = []
    if sec_run.m_xpath(
        f'method[{method_index}].force_field.model[{model_index}].contributions'
    ):
        contributions = (
            sec_run.method[method_index].force_field.model[model_index].contributions
        )
    bond_list = []
    for contribution in contributions:
        if contribution.type != 'bond':
            continue

        atom_indices = contribution.atom_indices
        if (
            contribution.n_interactions
        ):  # all bonds have been grouped into one contribution
            bond_list = [tuple(indices) for indices in atom_indices]
        else:
            bond_list.append(tuple(contribution.atom_indices))

    return bond_list


def create_empty_universe(
    n_atoms: int,
    n_frames: int = 1,
    n_residues: int = 1,
    n_segments: int = 1,
    atom_resindex: Optional[np.ndarray] = None,
    residue_segindex: Optional[np.ndarray] = None,
    flag_trajectory: bool = False,
    flag_velocities: bool = False,
    flag_forces: bool = False,
    timestep: Optional[float] = None,
) -> MDAnalysis.Universe:
    """Create a blank Universe

    This function was adapted from the function empty() within the MDA class Universe().
    The only difference is that the Universe() class is imported directly here, whereas in the
    original function is is passed as a function argument, since the function there is a classfunction.

    Useful for building a Universe without requiring existing files,
    for example for system building.

    If `flag_trajectory` is set to True, a
    :class:`MDAnalysis.coordinates.memory.MemoryReader` will be
    attached to the Universe.

    Parameters
    ----------
    n_atoms: int
      number of Atoms in the Universe
    n_residues: int, default 1
      number of Residues in the Universe, defaults to 1
    n_segments: int, default 1
      number of Segments in the Universe, defaults to 1
    atom_resindex: array like, optional
      mapping of atoms to residues, e.g. with 6 atoms,
      `atom_resindex=[0, 0, 1, 1, 2, 2]` would put 2 atoms
      into each of 3 residues.
    residue_segindex: array like, optional
      mapping of residues to segments
    flag_trajectory: bool, optional
      if True, attaches a :class:`MDAnalysis.coordinates.memory.MemoryReader`
      allowing coordinates to be set and written.  Default is False
    flag_velocities: bool, optional
      include velocities in the :class:`MDAnalysis.coordinates.memory.MemoryReader`
    flag_forces: bool, optional
      include forces in the :class:`MDAnalysis.coordinates.memory.MemoryReader`

    Returns
    -------
    MDAnalysis.Universe object

    Examples
    --------
    For example to create a new Universe with 6 atoms in 2 residues, with
    positions for the atoms and a mass attribute:

    >>> u = mda.Universe.empty(6, 2,
                                atom_resindex=np.array([0, 0, 0, 1, 1, 1]),
                                flag_trajectory=True,
            )
    >>> u.add_TopologyAttr('masses')

    .. versionadded:: 0.17.0
    .. versionchanged:: 0.19.0
        The attached Reader when flag_trajectory=True is now a MemoryReader
    .. versionchanged:: 1.0.0
        Universes can now be created with 0 atoms
    """

    if not n_atoms:
        n_residues = 0
        n_segments = 0

    if atom_resindex is None:
        LOGGER.warning(
            'Residues specified but no atom_resindex given.  '
            'All atoms will be placed in first Residue.',
        )

    if residue_segindex is None:
        LOGGER.warning(
            'Segments specified but no segment_resindex given.  '
            'All residues will be placed in first Segment',
        )

    topology = Topology(
        n_atoms,
        n_residues,
        n_segments,
        atom_resindex=atom_resindex,
        residue_segindex=residue_segindex,
    )

    universe = Universe(topology)

    if flag_trajectory:
        coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
        vels = np.zeros_like(coords) if flag_velocities else None
        forces = np.zeros_like(coords) if flag_forces else None

        # grab and attach a MemoryReader
        universe.trajectory = get_reader_for(coords)(
            coords,
            order='fac',
            n_atoms=n_atoms,
            velocities=vels,
            forces=forces,
            dt=timestep,
        )

    return universe


def archive_to_universe(
    archive,
    system_index: int = 0,
    method_index: int = -1,
    model_index: int = -1,
) -> MDAnalysis.Universe:
    """Extract the topology from a provided run section of an archive entry

    Input:

        archive_sec_run: section run of an EntryArchive

        system_index: list index of archive.run[].system to be used for topology extraction

        method_index: list index of archive.run[].method to be used for atom parameter (charges and masses) extraction

        model_index: list index of archive.run[].method[].force_field.model for bond list extraction

    Variables:

        n_frames (int):

        n_atoms (int):

        atom_names (str, shape=(n_atoms)):

        atom_types (str, shape=(n_atoms)):

        atom_resindex (str, shape=(n_atoms)):

        atom_segids (str, shape=(n_atoms)):

        n_segments (int): Segments correspond to a group of the same type of molecules.

        n_residues (int): The number of distinct residues (nb - individual molecules are also denoted as a residue).

        resnames (str, shape=(n_residues)): The name of each residue.

        residue_segindex (int, shape=(n_residues)): The segment index that each residue belongs to.

        residue_molnums (int, shape=(n_residues)): The molecule index that each residue belongs to.

        residue_moltypes (int, shape=(n_residues)): The molecule type of each residue.

        n_molecules (int):

        masses (float, shape=(n_atoms)):  atom masses, units = amu

        charges (float, shape=(n_atoms)): atom partial charges, units = e

        positions (float, shape=(n_frames,n_atoms,3)): atom positions

        velocities (float, shape=(n_frames,n_atoms,3)): atom velocities

        dimensions (float, shape=(n_frames,6)): box dimensions (nb - currently assuming a cubic box!)

        bonds (tuple, shape=([])): list of tuples with the atom indices of each bond
    """

    try:
        sec_run = archive.run[-1]
        sec_system = sec_run.system
        sec_system_top = sec_run.system[system_index]
        sec_atoms = sec_system_top.atoms
        sec_atoms_group = sec_system_top.atoms_group
        sec_calculation = sec_run.calculation
        sec_method = (
            sec_run.method[method_index] if sec_run.get('method') is not None else {}
        )
    except IndexError:
        LOGGER.warning(
            'Supplied indices or necessary sections do not exist in archive. Cannot build the MDA universe.'
        )
        return None

    n_atoms = sec_atoms.get('n_atoms')
    if n_atoms is None:
        LOGGER.warning('No atoms found in the archive. Cannot build the MDA universe.')
        return None

    n_frames = len(sec_system) if sec_system is not None else 1
    atom_names = sec_atoms.get('labels')
    model_atom_parameters = sec_method.get('atom_parameters')
    atom_types = (
        [atom.label for atom in model_atom_parameters]
        if model_atom_parameters
        else atom_names
    )
    atom_resindex = np.arange(n_atoms)
    atoms_segindices = np.empty(n_atoms)
    atom_segids = np.array(range(n_atoms), dtype='object')
    molecule_groups = sec_atoms_group
    n_segments = len(molecule_groups)

    n_residues = 0
    n_molecules = 0
    residue_segindex = []
    resnames = []
    residue_moltypes = []
    residue_min_atom_index = []
    residue_n_atoms = []
    molecule_n_res = []
    for mol_group_ind, mol_group in enumerate(molecule_groups):
        atoms_segindices[mol_group.atom_indices] = mol_group_ind
        atom_segids[mol_group.atom_indices] = mol_group.label
        molecules = mol_group.atoms_group if mol_group.atoms_group is not None else []
        for mol in molecules:
            monomer_groups = mol.atoms_group
            mol_res_counter = 0
            if monomer_groups:
                for mon_group in monomer_groups:
                    monomers = mon_group.atoms_group
                    for mon in monomers:
                        resnames.append(mon.label)
                        residue_segindex.append(mol_group_ind)
                        residue_moltypes.append(mol.label)
                        residue_min_atom_index.append(np.min(mon.atom_indices))
                        residue_n_atoms.append(len(mon.atom_indices))
                        n_residues += 1
                        mol_res_counter += 1
            else:  # no monomers => whole molecule is it's own residue
                resnames.append(mol.label)
                residue_segindex.append(mol_group_ind)
                residue_moltypes.append(mol.label)
                residue_min_atom_index.append(np.min(mol.atom_indices))
                residue_n_atoms.append(len(mol.atom_indices))
                n_residues += 1
                mol_res_counter += 1
            molecule_n_res.append(mol_res_counter)
            n_molecules += 1

    # reorder the residues by atom_indices
    residue_data = np.array(
        [
            [
                residue_min_atom_index[i],
                residue_n_atoms[i],
                residue_segindex[i],
                residue_moltypes[i],
                resnames[i],
            ]
            for i in range(len(residue_min_atom_index))
        ],
        dtype=object,
    )
    residue_data = np.array(sorted(residue_data, key=lambda x: x[0], reverse=False)).T
    residue_n_atoms = residue_data[1].astype(int)
    residue_segindex = residue_data[2].astype(int)
    residue_moltypes = residue_data[3]
    resnames = residue_data[4]
    res_index_counter = 0
    for i_residue, res_n_atoms in enumerate(residue_n_atoms):
        atom_resindex[res_index_counter : res_index_counter + res_n_atoms] = i_residue  # type: ignore
        res_index_counter += res_n_atoms
    residue_molnums = np.array(range(n_residues))
    mol_index_counter = 0
    for i_molecule, n_res in enumerate(molecule_n_res):
        residue_molnums[mol_index_counter : mol_index_counter + n_res] = i_molecule
        mol_index_counter += n_res

    # get the atom masses and charges

    masses = np.empty(n_atoms)
    charges = np.empty(n_atoms)
    atom_parameters = (
        sec_method.get('atom_parameters') if sec_method is not None else []
    )
    atom_parameters = atom_parameters if atom_parameters is not None else []

    for atom_ind, atom in enumerate(atom_parameters):
        if atom.get('mass'):
            masses[atom_ind] = ureg.convert(
                atom.mass.magnitude, atom.mass.units, ureg.amu
            )
        if atom.get('charge'):
            charges[atom_ind] = ureg.convert(
                atom.charge.magnitude, atom.charge.units, ureg.e
            )

    # get the atom positions, velocites, and box dimensions
    positions = np.empty(shape=(n_frames, n_atoms, 3))
    velocities = np.empty(shape=(n_frames, n_atoms, 3))
    dimensions = np.empty(shape=(n_frames, 6))
    for frame_ind, frame in enumerate(sec_system):
        sec_atoms_fr = frame.get('atoms')
        if sec_atoms_fr is not None:
            positions_frame = sec_atoms_fr.positions
            positions[frame_ind] = (
                ureg.convert(
                    positions_frame.magnitude, positions_frame.units, ureg.angstrom
                )
                if positions_frame is not None
                else None
            )
            velocities_frame = sec_atoms_fr.velocities
            velocities[frame_ind] = (
                ureg.convert(
                    velocities_frame.magnitude,
                    velocities_frame.units,
                    ureg.angstrom / ureg.picosecond,
                )
                if velocities_frame is not None
                else None
            )
            latt_vec_tmp = sec_atoms_fr.get('lattice_vectors')
            if latt_vec_tmp is not None:
                length_conversion = ureg.convert(
                    1.0, sec_atoms_fr.lattice_vectors.units, ureg.angstrom
                )
                dimensions[frame_ind] = [
                    sec_atoms_fr.lattice_vectors.magnitude[0][0] * length_conversion,
                    sec_atoms_fr.lattice_vectors.magnitude[1][1] * length_conversion,
                    sec_atoms_fr.lattice_vectors.magnitude[2][2] * length_conversion,
                    90,
                    90,
                    90,
                ]  # TODO: extend to non-cubic boxes

    # get the bonds  # TODO extend to multiple storage options for interactions
    bonds = sec_atoms.bond_list
    if bonds is None:
        bonds = get_bond_list_from_model_contributions(
            sec_run, method_index=-1, model_index=-1
        )

    # get the system times
    system_timestep = 1.0 * ureg.picosecond

    def approx(a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    system_times = [calc.time for calc in sec_calculation if calc.system_ref]
    if system_times:
        try:
            method = archive.workflow2.method
            system_timestep = (
                method.integration_timestep * method.coordinate_save_frequency
            )
        except Exception:
            LOGGER.warning(
                'Cannot find the system times. MDA universe will contain non-physical times and timestep.'
            )
    else:
        time_steps = [
            system_times[i_time] - system_times[i_time - 1]
            for i_time in range(1, len(system_times))
        ]
        if all(approx(time_steps[0], time_step) for time_step in time_steps):
            system_timestep = ureg.convert(
                time_steps[0].magnitude, ureg.second, ureg.picosecond
            )
        else:
            LOGGER.warning(
                'System times are not equally spaced. Cannot set system times in MDA universe.'
                ' MDA universe will contain non-physical times and timestep.'
            )

    system_timestep = ureg.convert(
        system_timestep, system_timestep._units, ureg.picoseconds
    )

    # create the Universe
    metainfo_universe = create_empty_universe(
        n_atoms,
        n_frames=n_frames,
        n_residues=n_residues,
        n_segments=n_segments,
        atom_resindex=np.array(atom_resindex),
        residue_segindex=np.array(residue_segindex),
        flag_trajectory=True,
        flag_velocities=True,
        timestep=system_timestep.magnitude,
    )

    # set the positions and velocities
    for frame_ind, frame in enumerate(metainfo_universe.trajectory):
        metainfo_universe.atoms.positions = positions[frame_ind]
        metainfo_universe.atoms.velocities = velocities[frame_ind]

    # add the atom attributes
    metainfo_universe.add_TopologyAttr('name', atom_names)
    metainfo_universe.add_TopologyAttr('type', atom_types)
    metainfo_universe.add_TopologyAttr('mass', masses)
    metainfo_universe.add_TopologyAttr('charge', charges)
    if n_segments != 0:
        metainfo_universe.add_TopologyAttr('segids', np.unique(atom_segids))
    if n_residues != 0:
        metainfo_universe.add_TopologyAttr('resnames', resnames)
        metainfo_universe.add_TopologyAttr('resids', np.unique(atom_resindex) + 1)
        metainfo_universe.add_TopologyAttr('resnums', np.unique(atom_resindex) + 1)
    if len(residue_molnums) > 0:
        metainfo_universe.add_TopologyAttr('molnums', residue_molnums)
    if len(residue_moltypes) > 0:
        metainfo_universe.add_TopologyAttr('moltypes', residue_moltypes)

    # add the box dimensions
    for frame_ind, frame in enumerate(metainfo_universe.trajectory):
        metainfo_universe.atoms.dimensions = dimensions[frame_ind]

    # add the bonds
    if hasattr(metainfo_universe, 'bonds'):
        LOGGER.warning('archive_to_universe() failed, universe already has bonds.')
        return None
    metainfo_universe.add_TopologyAttr('bonds', bonds)

    return metainfo_universe


def _get_molecular_bead_groups(
    universe: MDAnalysis.Universe, moltypes: list[str] = []
) -> dict[str, BeadGroup]:
    """
    Creates bead groups based on the molecular types as defined by the MDAnalysis universe.
    """
    # Input validation
    if universe is None:
        LOGGER.warning('Universe required to create beads.')
        return {}

    if not moltypes:
        atoms_moltypes = getattr(universe.atoms, 'moltypes', [])
        moltypes = np.unique(atoms_moltypes)
    bead_groups = {}
    for moltype in moltypes:
        ags_by_moltype = universe.select_atoms('moltype ' + moltype)
        if ags_by_moltype.n_atoms == 0:
            continue

        if ags_by_moltype.masses is not None:
            ags_by_moltype = ags_by_moltype[
                ags_by_moltype.masses > abs(1e-2)
            ]  # remove any virtual/massless sites (needed for, e.g., 4-bead water models)
        bead_groups[moltype] = BeadGroup(ags_by_moltype, compound='fragments')

    return bead_groups


def calc_molecular_rdf(
    universe: MDAnalysis.Universe,
    bead_groups: dict[str, BeadGroup],
    n_traj_split: int = 10,
    n_prune: int = 1,
    interval_indices=None,
    max_mols: int = 5000,
) -> dict[str, Any]:
    """
    Calculates the radial distribution functions between for each unique pair of
    molecule types as a function of their center of mass distance.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        The MDAnalysis universe object.
    bead_groups : dict[str, BeadGroup]
        Precomputed bead groups for the universe.
    n_traj_split : int
        Number of intervals to split trajectory into for averaging.
    n_prune : int
        Pruning parameter for frames.
    interval_indices : list or None
        2D array specifying the groups of the n_traj_split intervals to be averaged.
    max_mols : int
        Maximum number of molecules per bead group for calculating the rdf, for efficiency purposes.
    """
    # TODO 5k default for max_mols was set after > 50k was giving problems. Should do further testing to see where the appropriate limit should be set.
    if bead_groups is None or not bead_groups:
        LOGGER.warning('bead_groups required to calculate RDF.')
        return {}

    if (
        not universe
        or not universe.trajectory
        or universe.trajectory[0].dimensions is None
    ):
        LOGGER.warning('universe required to calculate RDF.')
        return {}

    n_frames = universe.trajectory.n_frames
    if n_frames < n_traj_split:
        n_traj_split = 1
        frames_start = np.array([0])
        frames_end = np.array([n_frames])
        n_frames_split = np.array([n_frames])
        interval_indices = [[0]]
    else:
        run_len = int(n_frames / n_traj_split)
        frames_start = np.arange(n_traj_split) * run_len
        frames_end = frames_start + run_len
        frames_end[-1] = n_frames
        n_frames_split = frames_end - frames_start
        if np.sum(n_frames_split) != n_frames:
            LOGGER.error(
                'Something went wrong with input parameters in calc_molecular_rdf().'
                'Radial distribution functions will not be calculated.'
            )
            return {}
        if not interval_indices:
            interval_indices = [[i] for i in range(n_traj_split)]

    if not bead_groups:
        return bead_groups
    moltypes = list(bead_groups.keys())
    del_list = [
        i_moltype
        for i_moltype, moltype in enumerate(moltypes)
        if len(bead_groups[moltype].positions) > max_mols
    ]
    moltypes = np.delete(moltypes, del_list).tolist()

    min_box_dimension = np.min(universe.trajectory[0].dimensions[:3])
    max_rdf_dist = min_box_dimension / 2
    n_bins = 200
    n_smooth = 2

    rdf_results: dict[str, Any] = {}
    rdf_results['n_smooth'] = n_smooth
    rdf_results['n_prune'] = n_prune
    rdf_results['type'] = 'molecular'
    rdf_results['types'] = []
    rdf_results['variables_name'] = []
    rdf_results['bins'] = []
    rdf_results['value'] = []
    rdf_results['frame_start'] = []
    rdf_results['frame_end'] = []
    for i, moltype_i in enumerate(moltypes):
        for j, moltype_j in enumerate(moltypes):
            if j > i:
                continue
            elif (
                i == j and bead_groups[moltype_i].positions.shape[0] == 1
            ):  # skip if only 1 mol in group
                continue

            if i == j:
                exclusion_block = (1, 1)  # remove self-distance
            else:
                exclusion_block = None
            pair_type = f'{moltype_i}-{moltype_j}'
            rdf_results_interval: dict[str, Any] = {}
            rdf_results_interval['types'] = []
            rdf_results_interval['variables_name'] = []
            rdf_results_interval['bins'] = []
            rdf_results_interval['value'] = []
            rdf_results_interval['frame_start'] = []
            rdf_results_interval['frame_end'] = []
            for i_interval in range(n_traj_split):
                rdf_results_interval['types'].append(pair_type)
                rdf_results_interval['variables_name'].append(['distance'])
                rdf = MDA_RDF.InterRDF(
                    bead_groups[moltype_i],
                    bead_groups[moltype_j],
                    range=(0, max_rdf_dist),
                    exclusion_block=exclusion_block,
                    nbins=n_bins,
                ).run(frames_start[i_interval], frames_end[i_interval], n_prune)
                rdf_results_interval['frame_start'].append(frames_start[i_interval])
                rdf_results_interval['frame_end'].append(frames_end[i_interval])

                rdf_results_interval['bins'].append(
                    rdf.results.bins[int(n_smooth / 2) : -int(n_smooth / 2)]
                    * ureg.angstrom
                )
                rdf_results_interval['value'].append(
                    np.convolve(
                        rdf.results.rdf, np.ones((n_smooth,)) / n_smooth, mode='same'
                    )[int(n_smooth / 2) : -int(n_smooth / 2)]
                )

            flag_logging_error = False
            for interval_group in interval_indices:
                split_weights = n_frames_split[np.array(interval_group)] / np.sum(
                    n_frames_split[np.array(interval_group)]
                )
                if abs(np.sum(split_weights) - 1.0) > 1e-6:
                    flag_logging_error = True
                    continue
                rdf_values_avg = (
                    split_weights[0] * rdf_results_interval['value'][interval_group[0]]
                )
                for i_interval, interval in enumerate(interval_group[1:]):
                    if (
                        rdf_results_interval['types'][interval]
                        != rdf_results_interval['types'][interval - 1]
                    ):
                        flag_logging_error = True
                        continue
                    if (
                        rdf_results_interval['variables_name'][interval]
                        != rdf_results_interval['variables_name'][interval - 1]
                    ):
                        flag_logging_error = True
                        continue
                    if not (
                        rdf_results_interval['bins'][interval]
                        == rdf_results_interval['bins'][interval - 1]
                    ).all():
                        flag_logging_error = True
                        continue
                    rdf_values_avg += (
                        split_weights[i_interval + 1]
                        * rdf_results_interval['value'][interval]
                    )
                if flag_logging_error:
                    LOGGER.error(
                        'Something went wrong in calc_molecular_rdf(). Some interval groups were skipped.'
                    )
                rdf_results['types'].append(
                    rdf_results_interval['types'][interval_group[0]]
                )
                rdf_results['variables_name'].append(
                    rdf_results_interval['variables_name'][interval_group[0]]
                )
                rdf_results['bins'].append(
                    rdf_results_interval['bins'][interval_group[0]]
                )
                rdf_results['value'].append(rdf_values_avg)
                rdf_results['frame_start'].append(
                    int(rdf_results_interval['frame_start'][interval_group[0]])
                )
                rdf_results['frame_end'].append(
                    int(rdf_results_interval['frame_end'][interval_group[-1]])
                )

    return rdf_results


def __log_indices(first: int, last: int, num: int = 100):
    ls = np.logspace(0, np.log10(last - first + 1), num=num)
    return np.unique(np.int_(ls) - 1 + first)


def __correlation(function, positions: list[float]):
    iterator = iter(positions)
    start_frame = next(iterator)
    return map(lambda f: function(start_frame, f), chain([start_frame], iterator))


def _calc_diffusion_constant(
    times: np.ndarray, values: np.ndarray, dim: int = 3
) -> tuple[float, float]:
    """
    Determines the diffusion constant from a fit of the mean squared displacement
    vs. time according to the Einstein relation.
    """
    linear_model = linregress(times, values)
    slope = linear_model.slope
    error = linear_model.rvalue
    return slope * 1 / (2 * dim), error


def shifted_correlation_average(
    function: Callable,
    times: np.ndarray,
    positions: np.ndarray,
    index_distribution: Callable = __log_indices,
    correlation: Callable = __correlation,
    segments: int = 10,
    window: float = 0.5,
    skip: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Code adapted from MDevaluate module: https://github.com/mdevaluate/mdevaluate.git

    Calculate the time series for a correlation function.

    The times at which the correlation is calculated are determined automatically by the
    function given as ``index_distribution``. The default is a logarithmic distribution.

    The function has been edited so that the average is always calculated, i.e., average=True below.

    Args:
        function:   The function that should be correlated
        positions:     The coordinates of the simulation data
        index_distribution (opt.):
                    A function that returns the indices for which the timeseries
                    will be calculated
        correlation (function, opt.):
                    The correlation function
        segments (int, opt.):
                    The number of segments the time window will be shifted
        window (float, opt.):
                    The fraction of the simulation the time series will cover
        skip (float, opt.):
                    The fraction of the trajectory that will be skipped at the beginning,
                    if this is None the start index of the frames slice will be used,
                    which defaults to 0.
        counter (bool, opt.):
                    If True, returns length of frames (in general number of particles specified)
        average (bool, opt.):
                    If True,
    Returns:
        tuple:
            A list of length N that contains the indices of the frames at which
            the time series was calculated and a numpy array of shape (segments, N)
            that holds the (non-avaraged) correlation data

            if has_counter == True: adds number of counts to output tupel.
                                    if average is returned it will be weighted.

    Example:
        Calculating the mean square displacement of a coordinates object named ``coords``:

        >>> indices, data = shifted_correlation(msd, coords)
    """
    if window + skip >= 1:
        LOGGER.warning(
            'Invalid parameters for shifted_correlation(), resetting to defaults.',
        )
        window = 0.5
        skip = 0

    start_frames = np.unique(
        np.linspace(
            len(positions) * skip,
            len(positions) * (1 - window),
            num=segments,
            endpoint=False,
            dtype=int,
        )
    )
    num_frames = int(len(positions) * (window))

    idx = index_distribution(0, num_frames)

    def correlate(start_frame):
        shifted_idx = idx + start_frame
        return correlation(function, map(positions.__getitem__, shifted_idx))

    correlation_times = np.array([times[i] for i in idx]) - times[0]

    result: np.ndarray
    for i_start_frame, start_frame in enumerate(start_frames):
        if i_start_frame == 0:
            result = np.array(list(correlate(start_frame)))
        else:
            result += np.array(list(correlate(start_frame)))
    result = np.array(result)
    result = result / len(start_frames)

    return correlation_times, result


def calc_molecular_mean_squared_displacements(
    universe: MDAnalysis.Universe,
    bead_groups: dict[str, BeadGroup],
    max_mols: int = 5000,
) -> dict[str, Any]:
    """
    Calculates the mean squared displacement for the center of mass of each
    molecule type.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        The MDAnalysis universe object.
    bead_groups : dict[str, BeadGroup]
        Precomputed bead groups for the universe.
    max_mols : int
        Maximum number of molecules per bead group for calculating the msd, for efficiency purposes.
    """

    def parse_jumps(
        universe: MDAnalysis.Universe, selection: MDAnalysis.AtomGroup
    ):  # TODO Add output declaration
        """
        See __get_nojump_positions().
        """
        __ = universe.trajectory[0]
        prev = np.array(selection.positions)
        box = universe.trajectory[0].dimensions[:3]
        sparse_data = namedtuple('SparseData', ['data', 'row', 'col'])  # type: ignore[name-match]
        jump_data = (
            sparse_data(data=array('b'), row=array('l'), col=array('l')),
            sparse_data(data=array('b'), row=array('l'), col=array('l')),
            sparse_data(data=array('b'), row=array('l'), col=array('l')),
        )

        for i_frame, _ in enumerate(universe.trajectory[1:]):
            curr = np.array(selection.positions)
            delta = ((curr - prev) / box).round().astype(np.int8)
            prev = np.array(curr)
            for d in range(3):
                (col,) = np.where(delta[:, d] != 0)
                jump_data[d].col.extend(col)
                jump_data[d].row.extend([i_frame] * len(col))
                jump_data[d].data.extend(delta[col, d])

        return jump_data

    def generate_nojump_matrices(
        universe: MDAnalysis.Universe, selection: MDAnalysis.AtomGroup
    ):  # TODO Add output declaration
        """
        See __get_nojump_positions().
        """
        jump_data = parse_jumps(universe, selection)
        n_frames = len(universe.trajectory)
        n_atoms = selection.positions.shape[0]

        nojump_matrices = tuple(
            sparse.csr_matrix(
                (np.array(m.data), (m.row, m.col)), shape=(n_frames, n_atoms)
            )
            for m in jump_data
        )
        return nojump_matrices

    def get_nojump_positions(
        universe: MDAnalysis.Universe, selection: MDAnalysis.AtomGroup
    ) -> np.ndarray:
        """
        Unwraps the positions to create a continuous trajectory without jumps across periodic boundaries.
        """
        nojump_matrices = generate_nojump_matrices(universe, selection)
        box = universe.trajectory[0].dimensions[:3]

        nojump_positions = []
        for i_frame, __ in enumerate(universe.trajectory):
            delta = (
                np.array(
                    np.vstack([m[:i_frame, :].sum(axis=0) for m in nojump_matrices]).T
                )
                * box
            )
            nojump_positions.append(selection.positions - delta)

        return np.array(nojump_positions)

    def mean_squared_displacement(start: np.ndarray, current: np.ndarray):
        """
        Calculates mean square displacement between current and initial (start) coordinates.
        """
        vec = start - current
        return (vec**2).sum(axis=1).mean()

    if bead_groups is None or not bead_groups:
        LOGGER.warning('bead_groups required to calculate MSD.')
        return {}

    if (
        not universe
        or not universe.trajectory
        or universe.trajectory[0].dimensions is None
    ):
        LOGGER.warning('universe required to calculate MSD.')
        return {}

    n_frames = universe.trajectory.n_frames
    if n_frames < 50:
        LOGGER.warning(
            'At least 50 frames required to calculate molecular'
            ' mean squared displacements, skipping.',
        )
        return {}

    dt = getattr(universe.trajectory, 'dt')
    if dt is None:
        LOGGER.warning(
            'Universe is missing time step, cannot calculate molecular'
            ' mean squared displacements, skipping.',
        )
        return {}
    times = np.arange(n_frames) * dt

    if bead_groups is {}:
        return bead_groups

    moltypes = [moltype for moltype in bead_groups.keys()]
    del_list = []
    for i_moltype, moltype in enumerate(moltypes):
        if len(bead_groups[moltype].positions) > max_mols:
            if max_mols > 50000:
                LOGGER.warning(
                    'Calculating mean squared displacements for more than 50k molecules.'
                    ' Expect long processing times!',
                )
            try:
                # select max_mols nr. of rnd molecules from this moltype
                moltype_indices = np.array(
                    [atom._ix for atom in bead_groups[moltype]._atoms]
                )
                molnums = universe.atoms.molnums[moltype_indices]
                molnum_types = np.unique(molnums)
                molnum_types_rnd = np.sort(
                    np.random.choice(molnum_types, size=max_mols)
                )
                atom_indices_rnd = np.concatenate(
                    [moltype_indices[molnums == molnum] for molnum in molnum_types_rnd]
                )
                selection = ' '.join([str(i) for i in atom_indices_rnd])
                selection = f'index {selection}'
                ags_moltype_rnd = universe.select_atoms(selection)
                bead_groups[moltype] = BeadGroup(ags_moltype_rnd, compound='fragments')
                LOGGER.warning(
                    'Maximum number of molecules for calculating the msd has been reached.'
                    ' Will make a random selection for calculation.'
                )
            except Exception:
                LOGGER.warning(
                    'Error in selecting random molecules for large group when calculating msd. Skipping this molecule type.'
                )
                del_list.append(i_moltype)

    for index in sorted(del_list, reverse=True):
        del moltypes[index]

    msd_results: dict[str, Any] = {}
    msd_results['type'] = 'molecular'
    msd_results['direction'] = 'xyz'
    msd_results['value'] = []
    msd_results['times'] = []
    msd_results['diffusion_constant'] = []
    msd_results['error_diffusion_constant'] = []
    for moltype in moltypes:
        positions = get_nojump_positions(universe, bead_groups[moltype])
        results = shifted_correlation_average(
            mean_squared_displacement, times, positions
        )
        if results:
            msd_results['value'].append(results[1])
            msd_results['times'].append(results[0])
            diffusion_constant, error = _calc_diffusion_constant(*results)
            msd_results['diffusion_constant'].append(diffusion_constant)
            msd_results['error_diffusion_constant'].append(error)

    msd_results['types'] = moltypes
    msd_results['times'] = np.array(msd_results['times']) * ureg.picosecond
    msd_results['value'] = np.array(msd_results['value']) * ureg.angstrom**2
    msd_results['diffusion_constant'] = (
        np.array(msd_results['diffusion_constant']) * ureg.angstrom**2 / ureg.picosecond
    )
    msd_results['error_diffusion_constant'] = np.array(
        msd_results['error_diffusion_constant']
    )

    return msd_results


def calc_radius_of_gyration(
    universe: MDAnalysis.Universe, molecule_atom_indices: np.ndarray
) -> dict[str, Any]:
    """
    Calculates the radius of gyration as a function of time for the atoms 'molecule_atom_indices'.

    molecule_atom_indices : np.ndarray
        The indices of the atoms corresponding to a single molecule for which the Rg will be calculated.
    """
    if molecule_atom_indices is None or len(molecule_atom_indices) == 0:
        LOGGER.warning(
            'molecule_atom_indices is required to calculate radius of gyration'
        )
        return {}

    if (
        not universe
        or not universe.trajectory
        or universe.trajectory[0].dimensions is None
    ):
        LOGGER.warning('universe is None. Cannot calculate radius of gyration.')
        return {}
    selection = ' '.join([str(i) for i in molecule_atom_indices])
    selection = f'index {selection}'
    molecule = universe.select_atoms(selection)
    rg_results: dict[str, Any] = {}
    rg_results['type'] = 'molecular'
    rg_results['times'] = []
    rg_results['value'] = []
    time_unit = hasattr(universe.trajectory.time, 'units')
    for __ in universe.trajectory:
        rg_results['times'].append(
            universe.trajectory.time.magnitude
            if time_unit
            else universe.trajectory.time
        )
        rg_results['value'].append(molecule.radius_of_gyration())
    rg_results['n_frames'] = len(rg_results['times'])
    rg_results['times'] = (
        np.array(rg_results['times']) * time_unit
        if time_unit
        else np.array(rg_results['times'])
    )
    rg_results['value'] = np.array(rg_results['value']) * ureg.angstrom

    return rg_results


def calc_molecular_radius_of_gyration(
    universe: MDAnalysis.Universe, system_topology: MSection
) -> list[dict[str, Any]]:
    """
    Calculates the radius of gyration as a function of time for each polymer in the system.
    """
    if universe is None:
        LOGGER.warning('universe required to calculate molecular radius of gyration')
        return []
    if system_topology is None or not system_topology:
        LOGGER.warning(
            'system_topology require to calculate molecular radius of gyration.'
        )
        return []

    rg_results = []
    for molgroup in system_topology:
        for molecule in molgroup.get('atoms_group'):
            sec_monomer_groups = molecule.get('atoms_group')
            group_type = sec_monomer_groups[0].type if sec_monomer_groups else None
            if group_type != 'monomer_group':
                continue
            rg_result = calc_radius_of_gyration(universe, molecule.atom_indices)
            rg_result['label'] = molecule.label + '-index_' + str(molecule.index)
            rg_result['atomsgroup_ref'] = molecule
            rg_results.append(rg_result)

    return rg_results


def get_molecules_from_bond_list(
    n_particles: int,
    bond_list: list[tuple],
    particle_types: list[str] = [],
    particles_typeid: Optional[array] = None,
) -> list[dict[str, Any]]:
    """
    Returns a list of dictionaries with molecule info from each instance in the list of bonds.
    """
    system_graph = networkx.empty_graph(n_particles)
    system_graph.add_edges_from([(i[0], i[1]) for i in bond_list])
    molecules = [
        system_graph.subgraph(c).copy()
        for c in networkx.connected_components(system_graph)
    ]
    molecule_info: list[dict[str, Any]] = []
    molecule_dict: dict[str, Any] = {}
    for mol in molecules:
        molecule_dict = {}
        molecule_dict['indices'] = np.array(mol.nodes())
        molecule_dict['bonds'] = np.array(mol.edges())
        molecule_dict['type'] = 'molecule'
        molecule_dict['is_molecule'] = True
        if particles_typeid is None and len(particle_types) == n_particles:
            molecule_dict['names'] = [
                particle_types[int(x)]
                for x in sorted(np.array(molecule_dict['indices']))
            ]
        if particle_types is not None and particles_typeid is not None:
            molecule_dict['names'] = [
                particle_types[particles_typeid[int(x)]]
                for x in sorted(np.array(molecule_dict['indices']))
            ]
        molecule_info.append(molecule_dict)
    return molecule_info


def is_same_molecule(mol_1: dict, mol_2: dict) -> bool:
    """
    Checks whether the 2 input molecule dictionary (see "get_molecules_from_bond_list()" above)
    represent the same molecule type, i.e., same particle types and corresponding bond connections.
    """

    def get_bond_list_dict(mol):
        mol_shift = np.min(mol['indices'])
        mol_bonds_shift = mol['bonds'] - mol_shift
        bond_list = [
            sorted((mol['names'][i], mol['names'][j])) for i, j in mol_bonds_shift
        ]
        bond_list_names, bond_list_counts = np.unique(
            bond_list, axis=0, return_counts=True
        )

        return {
            bond[0] + '-' + bond[1]: bond_list_counts[i_bond]
            for i_bond, bond in enumerate(bond_list_names)
        }

    if sorted(mol_1['names']) != sorted(mol_2['names']):
        return False

    bond_list_dict_1 = get_bond_list_dict(mol_1)
    bond_list_dict_2 = get_bond_list_dict(mol_2)

    if bond_list_dict_1 == bond_list_dict_2:
        return True

    return False


def get_composition(children_names: list[str]) -> str:
    """
    Generates a generalized "chemical formula" based on the provided list `children_names`,
    with the format X(m)Y(n) for children_names X and Y of quantities m and n, respectively.
    """
    children_count_tup = np.unique(children_names, return_counts=True)
    formula = ''.join([f'{name}({count})' for name, count in zip(*children_count_tup)])
    return formula


def mda_universe_from_nomad_atoms(system, logger=None):
    """Returns an instance of mda.Universe from a NOMAD Atoms-section.

    Args:
        system: The atoms to transform

    Returns:
        A new mda.Universe created from the given data.
    """
    n_atoms = len(system.positions)
    n_residues = 1
    atom_resindex = [0] * n_atoms
    residue_segindex = [0]

    universe = MDAnalysis.Universe.empty(
        n_atoms,
        n_residues=n_residues,
        atom_resindex=atom_resindex,
        residue_segindex=residue_segindex,
        trajectory=True,
    )

    # Add positions
    universe.atoms.positions = system.positions.to(ureg.angstrom).magnitude

    # Add atom attributes
    atom_names = system.labels
    universe.add_TopologyAttr('name', atom_names)
    universe.add_TopologyAttr('type', atom_names)
    universe.add_TopologyAttr('element', atom_names)

    # Add the box dimensions
    if system.lattice_vectors is not None:
        universe.atoms.dimensions = atomutils.cell_to_cellpar(
            system.lattice_vectors.to(ureg.angstrom).magnitude, degrees=True
        )

    return universe


class ThermostatParameters(ArchiveSection):
    """
    Section containing the parameters pertaining to the thermostat for a molecular dynamics run.
    """

    m_def = Section(validate=False)

    thermostat_type = Quantity(
        type=MEnum(
            'andersen',
            'berendsen',
            'brownian',
            'dissipative_particle_dynamics',
            'langevin_goga',
            'langevin_leap_frog',
            'langevin_schneider',
            'nose_hoover',
            'velocity_rescaling',
            'velocity_rescaling_langevin',
            'velocity_rescaling_woodcock',
        ),
        shape=[],
        description="""
        The name of the thermostat used for temperature control. If skipped or an empty string is used, it
        means no thermostat was applied.

        Allowed values are:

        | Thermostat Name        | Description                               |

        | ---------------------- | ----------------------------------------- |

        | `""`                   | No thermostat               |

        | `"andersen"`           | H.C. Andersen, [J. Chem. Phys.
        **72**, 2384 (1980)](https://doi.org/10.1063/1.439486) |

        | `"berendsen"`          | H. J. C. Berendsen, J. P. M. Postma,
        W. F. van Gunsteren, A. DiNola, and J. R. Haak, [J. Chem. Phys.
        **81**, 3684 (1984)](https://doi.org/10.1063/1.448118) |

        | `"brownian"`           | Brownian Dynamics |

        | `"dissipative_particle_dynamics"` | R.D. Groot and P.B. Warren
        [J. Chem. Phys. **107**(11), 4423-4435 (1997)](https://doi.org/10.1063/1.474784) |

        | `"langevin_goga"`           | N. Goga, A. J. Rzepiela, A. H. de Vries,
        S. J. Marrink, and H. J. C. Berendsen, [J. Chem. Theory Comput. **8**, 3637 (2012)]
        (https://doi.org/10.1021/ct3000876) |

        | `"langevin_leap_frog"` | J.A. Izaguirre, C.R. Sweet, and V.S. Pande
        [Pac Symp Biocomput. **15**, 240-251 (2010)](https://doi.org/10.1142/9789814295291_0026) |

        | `"langevin_schneider"`           | T. Schneider and E. Stoll,
        [Phys. Rev. B **17**, 1302](https://doi.org/10.1103/PhysRevB.17.1302) |

        | `"nose_hoover"`        | S. Nosé, [Mol. Phys. **52**, 255 (1984)]
        (https://doi.org/10.1080/00268978400101201); W.G. Hoover, [Phys. Rev. A
        **31**, 1695 (1985) |

        | `"velocity_rescaling"` | G. Bussi, D. Donadio, and M. Parrinello,
        [J. Chem. Phys. **126**, 014101 (2007)](https://doi.org/10.1063/1.2408420) |

        | `"velocity_rescaling_langevin"` | G. Bussi and M. Parrinello,
        [Phys. Rev. E **75**, 056707 (2007)](https://doi.org/10.1103/PhysRevE.75.056707) |

        | `"velocity_rescaling_woodcock"` | L. V. Woodcock,
        [Chem. Phys. Lett. **10**, 257 (1971)](https://doi.org/10.1016/0009-2614(71)80281-6) |
        """,
    )

    reference_temperature = Quantity(
        type=np.float64,
        shape=[],
        unit='kelvin',
        description="""
        The target temperature for the simulation. Typically used when temperature_profile is "constant".
        """,
    )

    coupling_constant = Quantity(
        type=np.float64,
        shape=[],
        unit='s',
        description="""
        The time constant for temperature coupling. Need to describe what this means for the various
        thermostat options...
        """,
    )

    effective_mass = Quantity(
        type=np.float64,
        shape=[],
        unit='kilogram',
        description="""
        The effective or fictitious mass of the temperature resevoir.
        """,
    )

    temperature_profile = Quantity(
        type=MEnum('constant', 'linear', 'exponential'),
        shape=[],
        description="""
        Type of temperature control (i.e., annealing) procedure. Can be "constant" (no annealing), "linear", or "exponential".
        If linear, "temperature_update_delta" specifies the corresponding update parameter.
        If exponential, "temperature_update_factor" specifies the corresponding update parameter.
        """,
    )

    reference_temperature_start = Quantity(
        type=np.float64,
        shape=[],
        unit='kelvin',
        description="""
        The initial target temperature for the simulation. Typically used when temperature_profile is "linear" or "exponential".
        """,
    )

    reference_temperature_end = Quantity(
        type=np.float64,
        shape=[],
        unit='kelvin',
        description="""
        The final target temperature for the simulation.  Typically used when temperature_profile is "linear" or "exponential".
        """,
    )

    temperature_update_frequency = Quantity(
        type=int,
        shape=[],
        description="""
        Number of simulation steps between changing the target temperature.
        """,
    )

    temperature_update_delta = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Amount to be added (subtracted if negative) to the current reference_temperature
        at a frequency of temperature_update_frequency when temperature_profile is "linear".
        The reference temperature is then replaced by this new value until the next update.
        """,
    )

    temperature_update_factor = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Factor to be multiplied to the current reference_temperature at a frequency of temperature_update_frequency when temperature_profile is exponential.
        The reference temperature is then replaced by this new value until the next update.
        """,
    )

    step_start = Quantity(
        type=int,
        shape=[],
        description="""
        Trajectory step where this thermostating starts.
        """,
    )

    step_end = Quantity(
        type=int,
        shape=[],
        description="""
        Trajectory step number where this thermostating ends.
        """,
    )


class BarostatParameters(ArchiveSection):
    """
    Section containing the parameters pertaining to the barostat for a molecular dynamics run.
    """

    m_def = Section(validate=False)

    barostat_type = Quantity(
        type=MEnum(
            'berendsen',
            'martyna_tuckerman_tobias_klein',
            'nose_hoover',
            'parrinello_rahman',
            'stochastic_cell_rescaling',
        ),
        shape=[],
        description="""
        The name of the barostat used for temperature control. If skipped or an empty string is used, it
        means no barostat was applied.

        Allowed values are:

        | Barostat Name          | Description                               |

        | ---------------------- | ----------------------------------------- |

        | `""`                   | No thermostat               |

        | `"berendsen"`          | H. J. C. Berendsen, J. P. M. Postma,
        W. F. van Gunsteren, A. DiNola, and J. R. Haak, [J. Chem. Phys.
        **81**, 3684 (1984)](https://doi.org/10.1063/1.448118) |

        | `"martyna_tuckerman_tobias_klein"` | G.J. Martyna, M.E. Tuckerman, D.J. Tobias, and M.L. Klein,
        [Mol. Phys. **87**, 1117 (1996)](https://doi.org/10.1080/00268979600100761);
        M.E. Tuckerman, J. Alejandre, R. López-Rendón, A.L. Jochim, and G.J. Martyna,
        [J. Phys. A. **59**, 5629 (2006)](https://doi.org/10.1088/0305-4470/39/19/S18)|

        | `"nose_hoover"`        | S. Nosé, [Mol. Phys. **52**, 255 (1984)]
        (https://doi.org/10.1080/00268978400101201); W.G. Hoover, [Phys. Rev. A
        **31**, 1695 (1985) |

        | `"parrinello_rahman"`        | M. Parrinello and A. Rahman,
        [J. Appl. Phys. **52**, 7182 (1981)](https://doi.org/10.1063/1.328693);
        S. Nosé and M.L. Klein, [Mol. Phys. **50**, 1055 (1983) |

        | `"stochastic_cell_rescaling"` | M. Bernetti and G. Bussi,
        [J. Chem. Phys. **153**, 114107 (2020)](https://doi.org/10.1063/1.2408420) |
        """,
    )

    coupling_type = Quantity(
        type=MEnum('isotropic', 'semi_isotropic', 'anisotropic'),
        shape=[],
        description="""
        Describes the symmetry of pressure coupling. Specifics can be inferred from the `coupling constant`

        | Type          | Description                               |

        | ---------------------- | ----------------------------------------- |

        | `isotropic`          | Identical coupling in all directions. |

        | `semi_isotropic` | Identical coupling in 2 directions. |

        | `anisotropic`        | General case. |
        """,
    )

    reference_pressure = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='pascal',
        description="""
        The target pressure for the simulation, stored in a 3x3 matrix, indicating the values for individual directions
        along the diagonal, and coupling between directions on the off-diagonal. Typically used when pressure_profile is "constant".
        """,
    )

    coupling_constant = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='s',
        description="""
        The time constants for pressure coupling, stored in a 3x3 matrix, indicating the values for individual directions
        along the diagonal, and coupling between directions on the off-diagonal. 0 values along the off-diagonal
        indicate no-coupling between these directions.
        """,
    )

    compressibility = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='1 / pascal',
        description="""
        An estimate of the system's compressibility, used for box rescaling, stored in a 3x3 matrix indicating the values for individual directions
        along the diagonal, and coupling between directions on the off-diagonal. If None, it may indicate that these values
        are incorporated into the coupling_constant, or simply that the software used uses a fixed value that is not available in
        the input/output files.
        """,
    )

    pressure_profile = Quantity(
        type=MEnum('constant', 'linear', 'exponential'),
        shape=[],
        description="""
        Type of pressure control procedure. Can be "constant" (no annealing), "linear", or "exponential".
        If linear, "pressure_update_delta" specifies the corresponding update parameter.
        If exponential, "pressure_update_factor" specifies the corresponding update parameter.
        """,
    )

    reference_pressure_start = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='pascal',
        description="""
        The initial target pressure for the simulation, stored in a 3x3 matrix, indicating the values for individual directions
        along the diagonal, and coupling between directions on the off-diagonal. Typically used when pressure_profile is "linear" or "exponential".
        """,
    )

    reference_pressure_end = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='pascal',
        description="""
        The final target pressure for the simulation, stored in a 3x3 matrix, indicating the values for individual directions
        along the diagonal, and coupling between directions on the off-diagonal.  Typically used when pressure_profile is "linear" or "exponential".
        """,
    )

    pressure_update_frequency = Quantity(
        type=int,
        shape=[],
        description="""
        Number of simulation steps between changing the target pressure.
        """,
    )

    pressure_update_delta = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Amount to be added (subtracted if negative) to the current reference_pressure
        at a frequency of pressure_update_frequency when pressure_profile is "linear".
        The pressure temperature is then replaced by this new value until the next update.
        """,
    )

    pressure_update_factor = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Factor to be multiplied to the current reference_pressure at a frequency of pressure_update_frequency when pressure_profile is exponential.
        The reference pressure is then replaced by this new value until the next update.
        """,
    )

    step_start = Quantity(
        type=int,
        shape=[],
        description="""
        Trajectory step where this barostating starts.
        """,
    )

    step_end = Quantity(
        type=int,
        shape=[],
        description="""
        Trajectory step number where this barostating ends.
        """,
    )


class ShearParameters(ArchiveSection):
    """
    Section containing the parameters pertaining to the shear flow for a molecular dynamics run.
    """

    m_def = Section(validate=False)

    shear_type = Quantity(
        type=MEnum('lees_edwards', 'trozzi_ciccotti', 'ashurst_hoover'),
        shape=[],
        description="""
        The name of the method used to implement the effect of shear flow within the simulation.

        Allowed values are:

        | Shear Method          | Description                               |

        | ---------------------- | ----------------------------------------- |

        | `""`                   | No thermostat               |

        | `"lees_edwards"`          | A.W. Lees and S.F. Edwards,
        [J. Phys. C **5** (1972) 1921](https://doi.org/10.1088/0022-3719/5/15/006)|

        | `"trozzi_ciccotti"`          | A.W. Lees and S.F. Edwards,
        [Phys. Rev. A **29** (1984) 916](https://doi.org/10.1103/PhysRevA.29.916)|

        | `"ashurst_hoover"`          | W. T. Ashurst and W. G. Hoover,
        [Phys. Rev. A **11** (1975) 658](https://doi.org/10.1103/PhysRevA.11.658)|
        """,
    )

    shear_rate = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='ps^-1',
        description="""
        The external stress tensor include normal (diagonal elements; which are zero in shear simulations)
        and shear stress' rates (off-diagonal elements).
        Its elements are: [[σ_x, τ_yx, τ_zx], [τ_xy, σ_y, τ_zy], [τ_xz, τ_yz, σ_z]],
		where σ and τ are the normal and shear stress' rates.
        The first and second letters in the index correspond to the normal vector to the shear plane and the direction of shearing, respectively.
        """,
    )

    step_start = Quantity(
        type=int,
        shape=[],
        description="""
        Trajectory step where this shearing starts.
        """,
    )

    step_end = Quantity(
        type=int,
        shape=[],
        description="""
        Trajectory step number where this shearing ends.
        """,
    )


class Lambdas(ArchiveSection):
    """
    Section for storing all lambda parameters for free energy perturbation
    """

    m_def = Section(validate=False)

    type = Quantity(
        type=MEnum(
            'output', 'coulomb', 'vdw', 'bonded', 'restraint', 'mass', 'temperature'
        ),
        shape=[],
        description="""
        The type of lambda interpolation

        Allowed values are:

        | type          | Description                               |

        | ---------------------- | ----------------------------------------- |

        | `"output"`           | Lambdas for the free energy outputs saved.
                                    These will also act as a default in case some
                                    relevant lambdas are not specified. |

        | `"coulomb"`          | Lambdas for interpolating electrostatic interactions. |

        | `"vdw"`              | Lambdas for interpolating van der Waals interactions. |

        | `"bonded"`           | Lambdas for interpolating all intramolecular interactions. |

        | `"restraint"`        | Lambdas for interpolating restraints. |

        | `"mass"`             | Lambdas for interpolating masses. |

        | `"temperature"`      | Lambdas for interpolating temperature. |
        """,
    )

    value = Quantity(
        type=np.float64,
        shape=[],
        description="""
        List of lambdas.
        """,
    )


class FreeEnergyCalculationParameters(ArchiveSection):
    """
    Section containing the parameters pertaining to a free energy calculation workflow
    that interpolates between two system states (defined via the interpolation parameter lambda).
    The parameters are stored for each molecular dynamics run separately, to be referenced
    by the overarching workflow.
    """

    m_def = Section(validate=False)

    type = Quantity(
        type=MEnum('alchemical', 'umbrella_sampling'),
        shape=[],
        description="""
        Specifies the type of workflow. Allowed values are:

        | kind          | Description                               |

        | ---------------------- | ----------------------------------------- |

        | `"alchemical"`           | A non-physical transformation between 2 well-defined systems,
                                     typically achieved by smoothly interpolating between Hamiltonians or force fields.  |

        | `"umbrella_sampling"`    | A sampling of the path between 2 well-defined (sub)states of a system,
                                     typically achieved by applying a biasing force to the force field along a
                                     specified reaction coordinate.
        """,
    )

    lambdas = SubSection(
        sub_section=Lambdas.m_def,
        description="""
        Contains the lists of lambda values defined for the interpolation of the system.
        """,
        repeats=True,
    )

    lambda_index = Quantity(
        type=int,
        shape=[],
        description="""
        The index of the lambda in `lambdas` corresponding to the state of the current simulation.
        """,
    )

    atom_indices = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description="""
        List of atom indices involved in the interpolation.
        """,
    )

    initial_state_vdw = Quantity(
        type=bool,
        shape=[],
        description="""
        Specifies whether vdw interactions are on (True) or off (False) in the initial state (i.e., lambda = 0).
        """,
    )

    final_state_vdw = Quantity(
        type=bool,
        shape=[],
        description="""
        Specifies whether vdw interactions are on (True) or off (False) in the final state (i.e., lambda = 0).
        """,
    )

    initial_state_coloumb = Quantity(
        type=bool,
        shape=[],
        description="""
        Specifies whether vdw interactions are on (True) or off (False) in the initial state (i.e., lambda = 0).
        """,
    )

    final_state_coloumb = Quantity(
        type=bool,
        shape=[],
        description="""
        Specifies whether vdw interactions are on (True) or off (False) in the final state (i.e., lambda = 0).
        """,
    )

    initial_state_bonded = Quantity(
        type=bool,
        shape=[],
        description="""
        Specifies whether bonded interactions are on (True) or off (False) in the initial state (i.e., lambda = 0).
        """,
    )

    final_state_bonded = Quantity(
        type=bool,
        shape=[],
        description="""
        Specifies whether bonded interactions are on (True) or off (False) in the final state (i.e., lambda = 0).
        """,
    )


class MolecularDynamicsMethod(SimulationWorkflowMethod):
    thermodynamic_ensemble = Quantity(
        type=MEnum('NVE', 'NVT', 'NPT', 'NPH'),
        shape=[],
        description="""
        The type of thermodynamic ensemble that was simulated.

        Allowed values are:

        | Thermodynamic Ensemble          | Description                               |

        | ---------------------- | ----------------------------------------- |

        | `"NVE"`           | Constant number of particles, volume, and energy |

        | `"NVT"`           | Constant number of particles, volume, and temperature |

        | `"NPT"`           | Constant number of particles, pressure, and temperature |

        | `"NPH"`           | Constant number of particles, pressure, and enthalpy |
        """,
    )

    integrator_type = Quantity(
        type=MEnum(
            'brownian',
            'conjugant_gradient',
            'langevin_goga',
            'langevin_schneider',
            'leap_frog',
            'rRESPA_multitimescale',
            'velocity_verlet',
            'langevin_leap_frog',
        ),
        shape=[],
        description="""
        Name of the integrator.

        Allowed values are:

        | Integrator Name          | Description                               |

        | ---------------------- | ----------------------------------------- |

        | `"langevin_goga"`           | N. Goga, A. J. Rzepiela, A. H. de Vries,
        S. J. Marrink, and H. J. C. Berendsen, [J. Chem. Theory Comput. **8**, 3637 (2012)]
        (https://doi.org/10.1021/ct3000876) |

        | `"langevin_schneider"`           | T. Schneider and E. Stoll,
        [Phys. Rev. B **17**, 1302](https://doi.org/10.1103/PhysRevB.17.1302) |

        | `"leap_frog"`          | R.W. Hockney, S.P. Goel, and J. Eastwood,
        [J. Comp. Phys. **14**, 148 (1974)](https://doi.org/10.1016/0021-9991(74)90010-2) |

        | `"velocity_verlet"` | W.C. Swope, H.C. Andersen, P.H. Berens, and K.R. Wilson,
        [J. Chem. Phys. **76**, 637 (1982)](https://doi.org/10.1063/1.442716) |

        | `"rRESPA_multitimescale"` | M. Tuckerman, B. J. Berne, and G. J. Martyna
        [J. Chem. Phys. **97**, 1990 (1992)](https://doi.org/10.1063/1.463137) |

        | `"langevin_leap_frog"` | J.A. Izaguirre, C.R. Sweet, and V.S. Pande
        [Pac Symp Biocomput. **15**, 240-251 (2010)](https://doi.org/10.1142/9789814295291_0026) |
        """,
    )

    integration_timestep = Quantity(
        type=np.float64,
        shape=[],
        unit='s',
        description="""
        The timestep at which the numerical integration is performed.
        """,
    )

    n_steps = Quantity(
        type=int,
        shape=[],
        description="""
        Number of timesteps performed.
        """,
    )

    coordinate_save_frequency = Quantity(
        type=int,
        shape=[],
        description="""
        The number of timesteps between saving the coordinates.
        """,
    )

    velocity_save_frequency = Quantity(
        type=int,
        shape=[],
        description="""
        The number of timesteps between saving the velocities.
        """,
    )

    force_save_frequency = Quantity(
        type=int,
        shape=[],
        description="""
        The number of timesteps between saving the forces.
        """,
    )

    thermodynamics_save_frequency = Quantity(
        type=int,
        shape=[],
        description="""
        The number of timesteps between saving the thermodynamic quantities.
        """,
    )

    thermostat_parameters = SubSection(
        sub_section=ThermostatParameters.m_def, repeats=True
    )

    barostat_parameters = SubSection(sub_section=BarostatParameters.m_def, repeats=True)

    shear_parameters = SubSection(sub_section=ShearParameters.m_def, repeats=True)

    free_energy_calculation_parameters = SubSection(
        sub_section=FreeEnergyCalculationParameters.m_def, repeats=True
    )


class Property(ArchiveSection):
    """
    Generic parent section for all property types.
    """

    m_def = Section(validate=False)

    type = Quantity(
        type=MEnum('molecular', 'atomic'),
        shape=[],
        description="""
        Describes if the observable is calculated at the molecular or atomic level.
        """,
    )

    label = Quantity(
        type=str,
        shape=[],
        description="""
        Name or description of the property.
        """,
    )

    error_type = Quantity(
        type=str,
        shape=[],
        description="""
        Describes the type of error reported for this observable.
        """,
    )


class PropertyValues(MSection):
    """
    Generic parent section for information regarding the values of a property.
    """

    m_def = Section(validate=False)

    label = Quantity(
        type=str,
        shape=[],
        description="""
        Describes the atoms or molecule types involved in determining the property.
        """,
    )

    errors = Quantity(
        type=np.float64,
        shape=['*'],
        description="""
        Error associated with the determination of the property.
        """,
    )


class EnsemblePropertyValues(PropertyValues):
    """
    Generic section containing information regarding the values of an ensemble property.
    """

    m_def = Section(validate=False)

    n_bins = Quantity(
        type=int,
        shape=[],
        description="""
        Number of bins.
        """,
    )

    frame_start = Quantity(
        type=int,
        shape=[],
        description="""
        Trajectory frame number where the ensemble averaging starts.
        """,
    )

    frame_end = Quantity(
        type=int,
        shape=[],
        description="""
        Trajectory frame number where the ensemble averaging ends.
        """,
    )

    bins_magnitude = Quantity(
        type=np.float64,
        shape=['n_bins'],
        description="""
        Values of the variable along which the property is calculated.
        """,
    )

    bins_unit = Quantity(
        type=str,
        shape=[],
        description="""
        Unit of the given bins, using UnitRegistry() notation.
        """,
    )

    value_magnitude = Quantity(
        type=np.float64,
        shape=['n_bins'],
        description="""
        Values of the property.
        """,
    )

    value_unit = Quantity(
        type=str,
        shape=[],
        description="""
        Unit of the property, using UnitRegistry() notation.
        """,
    )


class RadialDistributionFunctionValues(EnsemblePropertyValues):
    """
    Section containing information regarding the values of
    radial distribution functions (rdfs).
    """

    m_def = Section(validate=False)

    bins = Quantity(
        type=np.float64,
        shape=['n_bins'],
        unit='m',
        description="""
        Distances along which the rdf was calculated.
        """,
    )

    value = Quantity(
        type=np.float64,
        shape=['n_bins'],
        description="""
        Values of the property.
        """,
    )


class EnsembleProperty(Property):
    """
    Generic section containing information about a calculation of any static observable
    from a trajectory (i.e., from an ensemble average).
    """

    m_def = Section(validate=False)

    n_smooth = Quantity(
        type=int,
        shape=[],
        description="""
        Number of bins over which the running average was computed for
        the observable `values'.
        """,
    )

    n_variables = Quantity(
        type=int,
        shape=[],
        description="""
        Number of variables along which the property is determined.
        """,
    )

    variables_name = Quantity(
        type=str,
        shape=['n_variables'],
        description="""
        Name/description of the independent variables along which the observable is defined.
        """,
    )

    ensemble_property_values = SubSection(
        sub_section=EnsemblePropertyValues.m_def, repeats=True
    )


class RadialDistributionFunction(EnsembleProperty):
    """
    Section containing information about the calculation of
    radial distribution functions (rdfs).
    """

    m_def = Section(validate=False)

    _rdf_results = None

    radial_distribution_function_values = SubSection(
        sub_section=RadialDistributionFunctionValues.m_def, repeats=True
    )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if self._rdf_results:
            self.type = self._rdf_results.get('type')
            self.n_smooth = self._rdf_results.get('n_smooth')
            self.n_prune = self._rdf_results.get('n_prune')
            self.n_variables = 1
            self.variables_name = ['distance']
            for i_pair, pair_type in enumerate(self._rdf_results.get('types', [])):
                sec_rdf_values = self.m_create(RadialDistributionFunctionValues)
                sec_rdf_values.label = str(pair_type)
                sec_rdf_values.n_bins = len(
                    self._rdf_results.get('bins', [[]] * i_pair)[i_pair]
                )
                sec_rdf_values.bins = self._rdf_results.get('bins', [[]] * i_pair)[
                    i_pair
                ]
                sec_rdf_values.value = self._rdf_results.get('value', [[]] * i_pair)[
                    i_pair
                ]
                sec_rdf_values.frame_start = self._rdf_results.get(
                    'frame_start', [[]] * i_pair
                )[i_pair]
                sec_rdf_values.frame_end = self._rdf_results.get(
                    'frame_end', [[]] * i_pair
                )[i_pair]


class TrajectoryProperty(Property):
    """
    Generic section containing information about a calculation of any observable
    defined and stored at each individual frame of a trajectory.
    """

    m_def = Section(validate=False)

    n_frames = Quantity(
        type=int,
        shape=[],
        description="""
        Number of frames for which the observable is stored.
        """,
    )

    frames = Quantity(
        type=np.int32,
        shape=['n_frames'],
        description="""
        Frames for which the observable is stored.
        """,
    )

    times = Quantity(
        type=np.float64,
        shape=['n_frames'],
        unit='s',
        description="""
        Times for which the observable is stored.
        """,
    )

    value_magnitude = Quantity(
        type=np.float64,
        shape=['n_frames'],
        description="""
        Values of the property.
        """,
    )

    value_unit = Quantity(
        type=str,
        shape=[],
        description="""
        Unit of the property, using UnitRegistry() notation.
        """,
    )

    errors = Quantity(
        type=np.float64,
        shape=['*'],
        description="""
        Error associated with the determination of the property.
        """,
    )


# TODO Rg + TrajectoryPropery should be removed from workflow. All properties dependent on a single configuration should be store in calculation
class RadiusOfGyration(TrajectoryProperty):
    """
    Section containing information about the calculation of
    radius of gyration (Rg).
    """

    m_def = Section(validate=False)

    _rg_results = None

    atomsgroup_ref = Quantity(
        type=Reference(AtomsGroup.m_def),
        shape=[1],
        description="""
        References to the atoms_group section containing the molecule for which Rg was calculated.
        """,
    )

    value = Quantity(
        type=np.float64,
        shape=['n_frames'],
        unit='m',
        description="""
        Values of the property.
        """,
    )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if self._rg_results:
            self.type = self._rg_results.get('type')
            self.label = self._rg_results.get('label')
            # TODO Fix this assignment fails with TypeError
            try:
                self.atomsgroup_ref = [self._rg_results.get('atomsgroup_ref')]
            except Exception:
                pass
            self.n_frames = self._rg_results.get('n_frames')
            self.times = self._rg_results.get('times')
            self.value = self._rg_results.get('value')


class FreeEnergyCalculations(TrajectoryProperty):
    """
    Section containing information regarding the instantaneous (i.e., for a single configuration)
    values of free energies calculated via thermodynamic perturbation.
    The values stored are actually infinitesimal changes in the free energy, determined as derivatives
    of the Hamiltonian with respect to the coupling parameter (lambda) defining each state for the perturbation.
    """

    m_def = Section(validate=False)

    method_ref = Quantity(
        type=Reference(FreeEnergyCalculationParameters.m_def),
        shape=[],
        description="""
        Links the free energy results with the method parameters.
        """,
    )

    lambda_index = Quantity(
        type=int,
        shape=[],
        description="""
        Index of the lambda state for the present simulation within the free energy calculation workflow.
        I.e., lambda = method_ref.lambdas.values[lambda_index]
        """,
    )

    n_states = Quantity(
        type=int,
        shape=[],
        description="""
        Number of states defined for the interpolation of the system, as indicate in `method_ref`.
        """,
    )

    value_total_energy_magnitude = Quantity(
        type=HDF5Dataset,
        shape=[],
        unit='joule',
        description="""
        Value of the total energy for the present lambda state. The expected dimensions are ["n_frames"].
        This quantity is a reference to the data (file+path), which is stored in an HDF5 file for efficiency.
        """,
    )

    value_PV_energy_magnitude = Quantity(
        type=HDF5Dataset,
        shape=[],
        unit='joule',
        description="""
        Value of the pressure-volume energy (i.e., P*V) for the present lambda state. The expected dimensions are ["n_frames"].
        This quantity is a reference to the data (file+path), which is stored in an HDF5 file for efficiency.
        """,
    )

    value_total_energy_differences_magnitude = Quantity(
        type=HDF5Dataset,
        shape=[],
        unit='joule',
        description="""
        Values correspond to the difference in total energy between each specified lambda state
        and the reference state, which corresponds to the value of lambda of the current simulation.
        The expected dimensions are ["n_frames", "n_states"].
        This quantity is a reference to the data (file+path), which is stored in an HDF5 file for efficiency.
        """,
    )

    value_total_energy_derivative_magnitude = Quantity(
        type=HDF5Dataset,
        shape=[],
        unit='joule',  # TODO check this unit
        description="""
        Value of the derivative of the total energy with respect to lambda, evaluated for the current
        lambda state. The expected dimensions are ["n_frames"].
        This quantity is a reference to the data (file+path), which is stored in an HDF5 file for efficiency.
        """,
    )


class DiffusionConstantValues(PropertyValues):
    """
    Section containing information regarding the diffusion constants.
    """

    m_def = Section(validate=False)

    value = Quantity(
        type=np.float64,
        shape=[],
        unit='m^2/s',
        description="""
        Values of the diffusion constants.
        """,
    )

    error_type = Quantity(
        type=str,
        shape=[],
        description="""
        Describes the type of error reported for this observable.
        """,
    )


class CorrelationFunctionValues(PropertyValues):
    """
    Generic section containing information regarding the values of a correlation function.
    """

    m_def = Section(validate=False)

    n_times = Quantity(
        type=int,
        shape=[],
        description="""
        Number of times windows for the calculation of the correlation function.
        """,
    )

    times = Quantity(
        type=np.float64,
        shape=['n_times'],
        unit='s',
        description="""
        Time windows used for the calculation of the correlation function.
        """,
    )

    value_magnitude = Quantity(
        type=np.float64,
        shape=['n_times'],
        description="""
        Values of the property.
        """,
    )

    value_unit = Quantity(
        type=str,
        shape=[],
        description="""
        Unit of the property, using UnitRegistry() notation.
        """,
    )


class MeanSquaredDisplacementValues(CorrelationFunctionValues):
    """
    Section containing information regarding the values of a mean squared displacements (msds).
    """

    m_def = Section(validate=False)

    times = Quantity(
        type=np.float64,
        shape=['n_times'],
        unit='s',
        description="""
        Time windows used for the calculation of the msds.
        """,
    )

    value = Quantity(
        type=np.float64,
        shape=['n_times'],
        unit='m^2',
        description="""
        Mean squared displacement values.
        """,
    )

    errors = Quantity(
        type=np.float64,
        shape=['*'],
        description="""
        Error associated with the determination of the msds.
        """,
    )

    diffusion_constant = SubSection(
        sub_section=DiffusionConstantValues.m_def, repeats=False
    )


class CorrelationFunction(Property):
    """
    Generic section containing information about a calculation of any time correlation
    function from a trajectory.
    """

    m_def = Section(validate=False)

    direction = Quantity(
        type=MEnum('x', 'y', 'z', 'xy', 'yz', 'xz', 'xyz'),
        shape=[],
        description="""
        Describes the direction in which the correlation function was calculated.
        """,
    )

    correlation_function_values = SubSection(
        sub_section=CorrelationFunctionValues.m_def, repeats=True
    )


class MeanSquaredDisplacement(CorrelationFunction):
    """
    Section containing information about a calculation of any mean squared displacements (msds).
    """

    m_def = Section(validate=False)

    _msd_results = None

    mean_squared_displacement_values = SubSection(
        sub_section=MeanSquaredDisplacementValues.m_def, repeats=True
    )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if not self._msd_results:
            return

        self.type = self._msd_results.get('type')
        self.direction = self._msd_results.get('direction')
        for i_type, moltype in enumerate(self._msd_results.get('types', [])):
            sec_msd_values = self.m_create(MeanSquaredDisplacementValues)
            sec_msd_values.label = str(moltype)
            sec_msd_values.n_times = len(
                self._msd_results.get('times', [[]] * i_type)[i_type]
            )
            sec_msd_values.times = (
                self._msd_results['times'][i_type]
                if self._msd_results.get('times') is not None
                else []
            )
            sec_msd_values.value = (
                self._msd_results['value'][i_type]
                if self._msd_results.get('value') is not None
                else []
            )
            sec_diffusion = sec_msd_values.m_create(DiffusionConstantValues)
            sec_diffusion.value = (
                self._msd_results['diffusion_constant'][i_type]
                if self._msd_results.get('diffusion_constant') is not None
                else []
            )
            sec_diffusion.error_type = 'Pearson correlation coefficient'
            if self._msd_results.get('error_diffusion_constant') is not None:
                errors = self._msd_results['error_diffusion_constant'][i_type]
                sec_diffusion.errors = (
                    list(errors) if isinstance(errors, (list, np.ndarray)) else [errors]
                )


class MolecularDynamicsResults(ThermodynamicsResults):
    finished_normally = Quantity(
        type=bool,
        shape=[],
        description="""
        Indicates if calculation terminated normally.
        """,
    )

    n_steps = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of trajectory steps""",
    )

    trajectory = Quantity(
        type=Reference(System),
        shape=['n_steps'],
        description="""
        Reference to the system of each step in the trajectory.
        """,
    )

    radial_distribution_functions = SubSection(
        sub_section=RadialDistributionFunction.m_def, repeats=True
    )

    ensemble_properties = SubSection(sub_section=EnsembleProperty.m_def, repeats=True)

    correlation_functions = SubSection(
        sub_section=CorrelationFunction.m_def, repeats=True
    )

    radius_of_gyration = SubSection(sub_section=RadiusOfGyration, repeats=True)

    mean_squared_displacements = SubSection(
        sub_section=MeanSquaredDisplacement.m_def, repeats=True
    )

    free_energy_calculations = SubSection(
        sub_section=FreeEnergyCalculations.m_def, repeats=True
    )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        try:
            universe = archive_to_universe(archive)
        except Exception:
            universe = None
            logger.warning(
                'Could not convert archive to MDAnalysis Universe, skipping MD results normalization.'
            )

        if universe is None:
            return

        bead_groups = _get_molecular_bead_groups(universe)

        # calculate molecular radial distribution functions
        if not self.radial_distribution_functions:
            n_traj_split = (
                10  # number of intervals to split trajectory into for averaging
            )
            interval_indices = []  # 2D array specifying the groups of the n_traj_split intervals to be averaged
            # first 20% of trajectory
            interval_indices.append(np.arange(int(n_traj_split * 0.20)))
            # last 80% of trajectory
            interval_indices.append(np.arange(n_traj_split)[len(interval_indices[0]) :])
            # last 60% of trajectory
            interval_indices.append(
                np.arange(n_traj_split)[len(interval_indices[0]) * 2 :]
            )
            # last 40% of trajectory
            interval_indices.append(
                np.arange(n_traj_split)[len(interval_indices[0]) * 3 :]
            )

            n_prune = int(universe.trajectory.n_frames / len(archive.run[-1].system))
            rdf_results = calc_molecular_rdf(
                universe,
                bead_groups,
                n_traj_split=n_traj_split,
                n_prune=n_prune,
                interval_indices=interval_indices,
            )
            if rdf_results:
                sec_rdfs = RadialDistributionFunction()
                sec_rdfs._rdf_results = rdf_results
                sec_rdfs.normalize(archive, logger)
                self.radial_distribution_functions.append(sec_rdfs)

        # calculate the molecular mean squared displacements
        if not self.mean_squared_displacements:
            msd_results = calc_molecular_mean_squared_displacements(
                universe, bead_groups
            )
            if msd_results:
                sec_msds = MeanSquaredDisplacement()
                sec_msds._msd_results = msd_results
                sec_msds.normalize(archive, logger)
                self.mean_squared_displacements.append(sec_msds)

        # calculate radius of gyration for polymers
        try:
            sec_systems = archive.run[-1].system
            sec_system = sec_systems[0]
            sec_calc = archive.run[-1].calculation
            sec_calc = sec_calc if sec_calc is not None else []
        except Exception:
            return

        flag_rgs = False
        for calc in sec_calc:
            if calc.get('radius_of_gyration'):
                flag_rgs = True
                break  # TODO Should transfer Rg's to workflow results if they are already supplied in calculation

        if not flag_rgs:
            sec_rgs_calc = None
            system_topology = sec_system.atoms_group
            rg_results = calc_molecular_radius_of_gyration(universe, system_topology)
            for rg in rg_results:
                n_frames = rg.get('n_frames')
                if len(sec_systems) != n_frames:
                    logger.warning(
                        'Mismatch in length of system references in calculation and calculated Rg values.'
                        'Will not store Rg values under calculation section'
                    )
                    continue

                sec_rgs = RadiusOfGyration()
                sec_rgs._rg_results = rg
                sec_rgs.normalize(archive, logger)
                self.radius_of_gyration.append(sec_rgs)

                for calc in sec_calc:
                    if not calc.system_ref:
                        continue
                    sys_ind = calc.system_ref.m_parent_index
                    sec_rgs_calc = calc.radius_of_gyration
                    if not sec_rgs_calc:
                        sec_rgs_calc = calc.m_create(RadiusOfGyrationCalculation)
                        sec_rgs_calc.kind = rg.get('type')
                    else:
                        sec_rgs_calc = sec_rgs_calc[0]
                    sec_rg_values = sec_rgs_calc.m_create(
                        RadiusOfGyrationValuesCalculation
                    )

                    # TODO Fix this assignment fails with TypeError
                    try:
                        sec_rg_values.atomsgroup_ref = [rg.get('atomsgroup_ref')]
                    except Exception:
                        pass
                    sec_rg_values.label = rg.get('label')
                    sec_rg_values.value = rg.get('value')[sys_ind]


class MolecularDynamics(SerialSimulation):
    method = SubSection(sub_section=MolecularDynamicsMethod)

    results = SubSection(sub_section=MolecularDynamicsResults)

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if not self.method:
            self.method = MolecularDynamicsMethod()
            self.inputs.append(Link(name=WORKFLOW_METHOD_NAME, section=self.method))

        if not self.results:
            self.results = MolecularDynamicsResults()
            self.outputs.append(Link(name=WORKFLOW_RESULTS_NAME, section=self.results))

        if self.results.trajectory is None and self._systems:
            self.results.trajectory = self._systems

        SimulationWorkflowResults.normalize(self.results, archive, logger)
