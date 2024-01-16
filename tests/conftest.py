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
import numpy as np
from typing import List, Union
import pytest
from ase import Atoms as aseAtoms
import ase.build
import re

from nomad.units import ureg
from nomad.normalizing import normalizers
from nomad.utils import get_logger
from nomad.datamodel import EntryArchive, EntryMetadata
from nomad.datamodel.metainfo.simulation.run import Run, Program
from nomad.datamodel.metainfo.simulation.method import (
    Method,
    BasisSetContainer,
    BasisSet,
    Electronic,
    DFT,
    XCFunctional,
    Functional,
    Electronic,
    Smearing,
    Scf,
    GW,
    Photon,
    BSE,
    DMFT,
    AtomParameters,
    TB,
    Wannier,
    LatticeModelHamiltonian,
    HubbardKanamoriModel,
)
from nomad.datamodel.metainfo.simulation.calculation import (
    Calculation,
    Energy,
    EnergyEntry,
    Dos,
    DosValues,
    BandStructure,
    BandEnergies,
    RadiusOfGyration,
    RadiusOfGyrationValues,
    GreensFunctions,
    Spectra,
    ElectronicStructureProvenance,
)

from nomad.datamodel.metainfo.simulation.method import (
    Method,
    BasisSetContainer,
    BasisSet,
    Electronic,
    XCFunctional,
    Functional,
    DFT,
    GW,
)
from nomad.datamodel.metainfo.simulation.system import AtomsGroup, System, Atoms
from nomad.datamodel.metainfo.workflow import Link, TaskReference
from simulationworkflowschema.molecular_dynamics import (
    DiffusionConstantValues,
    MeanSquaredDisplacement,
    MeanSquaredDisplacementValues,
    RadialDistributionFunction,
    RadialDistributionFunctionValues,
)
from simulationworkflowschema.equation_of_state import EOSFit
from simulationworkflowschema import (
    MolecularDynamicsMethod,
    MolecularDynamicsResults,
    SinglePoint,
    GeometryOptimization,
    GeometryOptimizationMethod,
    Elastic,
    ElasticResults,
    MolecularDynamics,
    EquationOfState,
    EquationOfStateResults,
    DFTPlusGWMethod,
    DFTPlusGW as GWworkflow,
    DFTPlusTBPlusDMFTMethod,
    DFTPlusTBPlusDMFT as DMFTworkflow,
    PhotonPolarization,
    PhotonPolarizationMethod,
    PhotonPolarizationResults,
    XS as XSworkflow,
    DMFTPlusMaxEntMethod,
    DMFTPlusMaxEnt as MaxEntworkflow,
)


LOGGER = get_logger(__name__)


def run_normalize(entry_archive: EntryArchive) -> EntryArchive:
    for normalizer_class in normalizers:
        normalizer = normalizer_class(entry_archive)
        normalizer.normalize()
    return entry_archive


def get_template_computation() -> EntryArchive:
    """Returns a basic archive template for a computational calculation"""
    template = EntryArchive()
    run = template.m_create(Run)
    run.program = Program(name="VASP", version="4.6.35")
    system = run.m_create(System)
    system.atoms = Atoms(
        lattice_vectors=[
            [5.76372622e-10, 0.0, 0.0],
            [0.0, 5.76372622e-10, 0.0],
            [0.0, 0.0, 4.0755698899999997e-10],
        ],
        positions=[
            [2.88186311e-10, 0.0, 2.0377849449999999e-10],
            [0.0, 2.88186311e-10, 2.0377849449999999e-10],
            [0.0, 0.0, 0.0],
            [2.88186311e-10, 2.88186311e-10, 0.0],
        ],
        labels=["Br", "K", "Si", "Si"],
        periodic=[True, True, True],
    )
    scc = run.m_create(Calculation)
    scc.system_ref = system
    scc.energy = Energy(
        free=EnergyEntry(value=-1.5936767191492225e-18),
        total=EnergyEntry(value=-1.5935696296699573e-18),
        total_t0=EnergyEntry(value=-3.2126683561907e-22),
    )
    return template


def get_template_dft() -> EntryArchive:
    """Returns a basic archive template for a DFT calculation."""
    template = get_template_computation()
    run = template.run[-1]
    method = run.m_create(Method)
    method.electrons_representation = [
        BasisSetContainer(
            type="plane waves",
            scope=["wavefunction"],
            basis_set=[
                BasisSet(
                    type="plane waves",
                    scope=["valence"],
                )
            ],
        )
    ]
    method.electronic = Electronic(method="DFT")
    xc_functional = XCFunctional(exchange=[Functional(name="GGA_X_PBE")])
    method.dft = DFT(xc_functional=xc_functional)
    scc = run.calculation[-1]
    scc.method_ref = method
    template.workflow2 = GeometryOptimization()
    return template


def get_template_excited(type: str) -> EntryArchive:
    """Returns a basic archive template for a ExcitedState calculation."""
    template = get_template_computation()
    run = template.run[-1]
    method = run.m_create(Method)
    if type == "GW":
        method.gw = GW(type="G0W0")
    elif type == "Photon":
        photon = Photon(multipole_type="dipole")
        method.m_add_sub_section(Method.photon, photon)
    elif type == "BSE":
        method.bse = BSE(type="Singlet", solver="Lanczos-Haydock")
    scc = run.calculation[-1]
    scc.method_ref = method
    template.workflow2 = SinglePoint()
    return template


def get_template_tb_wannier() -> EntryArchive:
    """Returns a basic archive template for a TB calculation."""
    template = get_template_computation()
    run = template.run[-1]
    run.program = Program(name="Wannier90", version="3.1.0")
    method = run.m_create(Method)
    method_tb = method.m_create(TB)
    method_tb.name = "Wannier"
    method_tb.wannier = Wannier(is_maximally_localized=False)
    system = run.system[-1]
    system.m_add_sub_section(
        System.atoms_group,
        AtomsGroup(
            label="Br",
            type="projection",
            index=0,
            is_molecule=False,
            n_atoms=1,
            atom_indices=np.array([0]),
        ),
    )
    scc = run.calculation[-1]
    scc.method_ref = method
    template.workflow2 = SinglePoint()
    return template


def get_template_dmft() -> EntryArchive:
    """Returns a basic archive template for a DMFT calculation."""
    template = get_template_computation()
    run = template.run[-1]
    run.program = Program(name="w2dynamics")
    input_method = run.m_create(Method)
    input_model = input_method.m_create(LatticeModelHamiltonian)
    input_model.hubbard_kanamori_model.append(
        HubbardKanamoriModel(orbital="d", u=4.0e-19, jh=0.6e-19)
    )
    method_dmft = run.m_create(Method)
    method_dmft.dmft = DMFT(
        impurity_solver="CT-HYB",
        n_impurities=1,
        n_electrons=[1.0],
        n_correlated_orbitals=[3.0],
        inverse_temperature=60.0,
        magnetic_state="paramagnetic",
    )
    method_dmft.starting_method_ref = input_method
    scc = run.calculation[-1]
    scc.method_ref = method_dmft
    template.workflow2 = SinglePoint()
    return template


def get_template_maxent() -> EntryArchive:
    """Returns a basic archive template for a MaxEnt analytical continuation calculation."""
    # TODO update when MaxEnt methodology is defined
    template = get_template_computation()
    run = template.run[-1]
    run.program = Program(name="w2dynamics")
    method = run.m_create(Method)
    scc = run.calculation[-1]
    scc.method_ref = method
    template.workflow2 = SinglePoint()
    return template


def get_template_for_structure(atoms: aseAtoms) -> EntryArchive:
    template = get_template_dft()
    template.run[0].calculation[0].system_ref = None
    template.run[0].calculation[0].eigenvalues.append(BandEnergies())
    template.run[0].calculation[0].eigenvalues[0].kpoints = [[0, 0, 0]]
    template.run[0].system = None

    # Fill structural information
    # system = template.run[0].m_create(System)
    # system.atom_positions = atoms.get_positions() * 1E-10
    # system.atom_labels = atoms.get_chemical_symbols()
    # system.simulation_cell = atoms.get_cell() * 1E-10
    # system.configuration_periodic_dimensions = atoms.get_pbc()
    system = get_section_system(atoms)
    template.run[0].m_add_sub_section(Run.system, system)

    return run_normalize(template)


def get_section_system(atoms: aseAtoms):
    system = System()
    system.atoms = aseAtoms(
        positions=atoms.get_positions() * 1e-10,
        labels=atoms.get_chemical_symbols(),
        lattice_vectors=atoms.get_cell() * 1e-10,
        periodic=atoms.get_pbc(),
    )
    return system


def add_template_dos(
    template: EntryArchive,
    fill: List = [[[0, 1], [2, 3]]],
    energy_reference_fermi: Union[float, None] = None,
    energy_reference_highest_occupied: Union[float, None] = None,
    energy_reference_lowest_unoccupied: Union[float, None] = None,
    n_values: int = 101,
    type: str = "electronic",
) -> EntryArchive:
    """Used to create a test data for DOS.

    Args:
        fill: List containing the energy ranges (eV) that should be filled with
            non-zero values, e.g. [[[0, 1], [2, 3]]]. Defaults to single channel DOS
            with a gap.
        energy_fermi: Fermi energy (eV)
        energy_reference_highest_occupied: Highest occupied energy (eV) as given by a parser.
        energy_reference_lowest_unoccupied: Lowest unoccupied energy (eV) as given by a parser.
        type: 'electronic' or 'vibrational'
        has_references: Whether the DOS has energy references or not.
    """
    if len(fill) > 1 and type != "electronic":
        raise ValueError("Cannot create spin polarized DOS for non-electronic data.")
    scc = template.run[0].calculation[0]
    dos_type = (
        Calculation.dos_electronic if type == "electronic" else Calculation.dos_phonon
    )
    energies = np.linspace(-5, 5, n_values)
    for i, range_list in enumerate(fill):
        dos = scc.m_create(Dos, dos_type)
        dos.spin_channel = i if (len(fill) == 2 and type == "electronic") else None
        dos.energies = energies * ureg.electron_volt
        dos_total = dos.m_create(DosValues, Dos.total)
        dos_value = np.zeros(n_values)
        for r in range_list:
            idx_bottom = (np.abs(energies - r[0])).argmin()
            idx_top = (np.abs(energies - r[1])).argmin()
            dos_value[idx_bottom:idx_top] = 1
        dos_total.value = dos_value

    if energy_reference_fermi is not None:
        energy_reference_fermi *= ureg.electron_volt
    if energy_reference_highest_occupied is not None:
        energy_reference_highest_occupied *= ureg.electron_volt
    if energy_reference_lowest_unoccupied is not None:
        energy_reference_lowest_unoccupied *= ureg.electron_volt
    scc.energy = Energy(
        fermi=energy_reference_fermi,
        highest_occupied=energy_reference_highest_occupied,
        lowest_unoccupied=energy_reference_lowest_unoccupied,
    )
    return template


def get_template_dos(
    fill: List = [[[0, 1], [2, 3]]],
    energy_reference_fermi: Union[float, None] = None,
    energy_reference_highest_occupied: Union[float, None] = None,
    energy_reference_lowest_unoccupied: Union[float, None] = None,
    n_values: int = 101,
    type: str = "electronic",
    normalize: bool = True,
) -> EntryArchive:
    archive = get_template_dft()
    archive = add_template_dos(
        archive,
        fill,
        energy_reference_fermi,
        energy_reference_highest_occupied,
        energy_reference_lowest_unoccupied,
        n_values,
        type,
    )
    if normalize:
        archive = run_normalize(archive)
    return archive


def add_template_band_structure(
    template: EntryArchive,
    band_gaps: List = None,
    type: str = "electronic",
    has_references: bool = True,
    has_reciprocal_cell: bool = True,
) -> EntryArchive:
    """Used to create a test data for band structures.

    Args:
        band_gaps: List containing the band gap value and band gap type as a
            tuple, e.g. [(1, 'direct'), (0.5, 'indirect)]. Band gap values are
            in eV. Use a list of Nones if you don't want a gap for a specific
            channel.
        type: 'electronic' or 'vibrational'
        has_references: Whether the band structure has energy references or not.
        has_reciprocal_cell: Whether the reciprocal cell is available or not.
    """
    if band_gaps is None:
        band_gaps = [None]
    if not has_reciprocal_cell:
        template.run[0].system[0].atoms = None
    scc = template.run[0].calculation[0]
    if type == "electronic":
        bs = scc.m_create(BandStructure, Calculation.band_structure_electronic)
        n_spin_channels = len(band_gaps)
        fermi: List[float] = []
        highest: List[float] = []
        lowest: List[float] = []
        for gap in band_gaps:
            if gap is None:
                highest.append(0)
                lowest.append(0)
                fermi.append(0)
            else:
                fermi.append(1 * 1.60218e-19)

        if has_references:
            scc.energy = Energy(fermi=fermi[0])
            if len(highest) > 0:
                scc.energy.highest_occupied = highest[0]
            if len(lowest) > 0:
                scc.energy.lowest_unoccupied = lowest[0]
    else:
        bs = scc.m_create(BandStructure, Calculation.band_structure_phonon)
        n_spin_channels = 1
    n_segments = 2
    full_space = np.linspace(0, 2 * np.pi, 200)
    k, m = divmod(len(full_space), n_segments)
    space = list(
        (
            full_space[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
            for i in range(n_segments)
        )
    )
    for i_seg in range(n_segments):
        krange = space[i_seg]
        n_points = len(krange)
        seg = bs.m_create(BandEnergies)
        energies = np.zeros((n_spin_channels, n_points, 2))
        k_points = np.zeros((n_points, 3))
        k_points[:, 0] = np.linspace(0, 1, n_points)
        if type == "electronic":
            for i_spin in range(n_spin_channels):
                if band_gaps[i_spin] is not None:
                    if band_gaps[i_spin][1] == "direct":
                        energies[i_spin, :, 0] = -np.cos(krange)
                        energies[i_spin, :, 1] = np.cos(krange)
                    elif band_gaps[i_spin][1] == "indirect":
                        energies[i_spin, :, 0] = -np.cos(krange)
                        energies[i_spin, :, 1] = np.sin(krange)
                    else:
                        raise ValueError("Invalid band gap type")
                    energies[i_spin, :, 1] += 2 + band_gaps[i_spin][0]
                else:
                    energies[i_spin, :, 0] = -np.cos(krange)
                    energies[i_spin, :, 1] = np.cos(krange)
        else:
            energies[0, :, 0] = -np.cos(krange)
            energies[0, :, 1] = np.cos(krange)
        seg.energies = energies * 1.60218e-19
        seg.kpoints = k_points
    return template


def get_template_band_structure(
    band_gaps: List = None,
    type: str = "electronic",
    has_references: bool = True,
    has_reciprocal_cell: bool = True,
    normalize: bool = True,
) -> EntryArchive:
    archive = get_template_dft()
    archive = add_template_band_structure(
        archive, band_gaps, type, has_references, has_reciprocal_cell
    )
    if normalize:
        archive = run_normalize(archive)
    return archive


def add_template_greens_functions(template: EntryArchive) -> EntryArchive:
    """Used to create a test data for Greens functions."""
    scc = template.run[0].calculation[0]
    sec_gfs = scc.m_create(GreensFunctions)
    sec_gfs.matsubara_freq = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    sec_gfs.tau = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    n_atoms = 1
    n_spin = 2
    n_orbitals = 3
    n_iw = len(sec_gfs.matsubara_freq)
    self_energy_iw = [
        [
            [[w * 1j + o + s + a for w in range(n_iw)] for o in range(n_orbitals)]
            for s in range(n_spin)
        ]
        for a in range(n_atoms)
    ]
    sec_gfs.self_energy_iw = self_energy_iw
    sec_gfs.greens_function_tau = self_energy_iw
    return template


def get_template_gw_workflow() -> EntryArchive:
    """Returns a basic archive template for a GW workflow entry, composed of two main tasks:
    DFT GeometryOptimization and GW SinglePoint."""
    # Defining DFT and GW SinglePoint archives and adding band_structure and dos to them.
    archive_dft = get_template_dft()
    archive_gw = get_template_excited(type="GW")
    archive_dft = add_template_band_structure(archive_dft)
    archive_gw = add_template_band_structure(archive_gw)
    archive_dft = add_template_dos(archive_dft)
    archive_gw = add_template_dos(archive_gw)
    # Normalizing SinglePoint archives
    run_normalize(archive_dft)
    run_normalize(archive_gw)
    # Defining DFT and GW tasks for later the GW workflow
    task_dft = TaskReference(task=archive_dft.workflow2)
    task_dft.name = "DFT"
    task_dft.inputs = [
        Link(name="Input structure", section=archive_dft.run[-1].system[-1])
    ]
    task_dft.outputs = [
        Link(name="Output DFT calculation", section=archive_dft.run[-1].calculation[-1])
    ]
    task_gw = TaskReference(task=archive_gw.workflow2)
    task_gw.name = "GW"
    task_gw.inputs = [
        Link(name="Output DFT calculation", section=archive_dft.run[-1].calculation[-1])
    ]
    task_gw.outputs = [
        Link(name="Output GW calculation", section=archive_gw.run[-1].calculation[-1])
    ]
    # GW workflow entry (no need of creating Method nor Calculation)
    template = EntryArchive()
    run = template.m_create(Run)
    run.program = archive_dft.run[-1].program
    run.system = archive_dft.run[-1].system
    workflow = GWworkflow()
    workflow_method = DFTPlusGWMethod(
        gw_method_ref=archive_gw.run[-1].method[-1].gw,
        starting_point=archive_dft.run[-1].method[-1].dft.xc_functional,
        electrons_representation=archive_dft.run[-1]
        .method[-1]
        .electrons_representation[-1],
    )
    workflow.m_add_sub_section(GWworkflow.method, workflow_method)
    workflow.m_add_sub_section(
        GWworkflow.inputs,
        Link(name="Input structure", section=archive_dft.run[-1].system[-1]),
    )
    workflow.m_add_sub_section(
        GWworkflow.outputs,
        Link(name="Output GW calculation", section=archive_gw.run[-1].calculation[-1]),
    )
    workflow.m_add_sub_section(GWworkflow.tasks, task_dft)
    workflow.m_add_sub_section(GWworkflow.tasks, task_gw)
    template.workflow2 = workflow
    return template


def get_template_dmft_workflow() -> EntryArchive:
    # Defining Projection and DMFT SinglePoint archives and adding band_structure and greens_functions to them.
    archive_tb = get_template_tb_wannier()
    archive_dmft = get_template_dmft()
    archive_tb = add_template_band_structure(archive_tb)
    archive_dmft = add_template_greens_functions(archive_dmft)
    # Normalizing SinglePoint archives BEFORE defining the DMFT workflow entry
    run_normalize(archive_tb)
    run_normalize(archive_dmft)
    # Defining Projection and DMFT tasks for later the DMFT workflow
    task_proj = TaskReference(task=archive_tb.workflow2)
    task_proj.name = "Projection"
    task_proj.inputs = [
        Link(name="Input structure", section=archive_tb.run[-1].system[-1])
    ]
    task_proj.outputs = [
        Link(name="Output TB calculation", section=archive_tb.run[-1].calculation[-1])
    ]
    task_dmft = TaskReference(task=archive_dmft.workflow2)
    task_dmft.name = "DMFT"
    task_dmft.inputs = [
        Link(name="Output TB calculation", section=archive_tb.run[-1].calculation[-1])
    ]
    task_dmft.outputs = [
        Link(
            name="Output DMFT calculation", section=archive_dmft.run[-1].calculation[-1]
        )
    ]
    # DMFT workflow entry (no need of creating Method nor Calculation)
    template = EntryArchive()
    run = template.m_create(Run)
    run.program = archive_dmft.run[-1].program
    run.system = archive_tb.run[-1].system
    workflow = DMFTworkflow()
    workflow_method = DFTPlusTBPlusDMFTMethod(
        tb_method_ref=archive_tb.run[-1].method[-1].tb,
        dmft_method_ref=archive_dmft.run[-1].method[-1].dmft,
    )
    workflow.m_add_sub_section(DMFTworkflow.method, workflow_method)
    workflow.m_add_sub_section(
        DMFTworkflow.inputs,
        Link(name="Input structure", section=archive_tb.run[-1].system[-1]),
    )
    workflow.m_add_sub_section(
        DMFTworkflow.outputs,
        Link(
            name="Output DMFT calculation", section=archive_dmft.run[-1].calculation[-1]
        ),
    )
    workflow.m_add_sub_section(DMFTworkflow.tasks, task_proj)
    workflow.m_add_sub_section(DMFTworkflow.tasks, task_dmft)
    template.workflow2 = workflow
    return template


def get_template_maxent_workflow() -> EntryArchive:
    # Defining Projection and DMFT SinglePoint archives and adding band_structure and greens_functions to them.
    archive_dmft = get_template_dmft()
    archive_maxent = get_template_maxent()
    archive_dmft = add_template_greens_functions(archive_dmft)
    archive_maxent = add_template_greens_functions(archive_maxent)
    # Normalizing SinglePoint archives BEFORE defining the DMFT workflow entry
    run_normalize(archive_dmft)
    run_normalize(archive_maxent)
    # Defining Projection and DMFT tasks for later the DMFT workflow
    task_dmft = TaskReference(task=archive_dmft.workflow2)
    task_dmft.name = "DMFT"
    task_dmft.inputs = [
        Link(name="Input structure", section=archive_dmft.run[-1].system[-1])
    ]
    task_dmft.outputs = [
        Link(
            name="Output DMFT calculation", section=archive_dmft.run[-1].calculation[-1]
        )
    ]
    task_maxent = TaskReference(task=archive_dmft.workflow2)
    task_maxent.name = "MaxEnt Sigma"
    task_maxent.inputs = [
        Link(
            name="Output DMFT calculation", section=archive_dmft.run[-1].calculation[-1]
        )
    ]
    task_maxent.outputs = [
        Link(
            name="Output MaxEnt Sigma calculation",
            section=archive_maxent.run[-1].calculation[-1],
        )
    ]
    # DMFT workflow entry (no need of creating Method)
    template = EntryArchive()
    run = template.m_create(Run)
    run.program = archive_dmft.run[-1].program
    run.system = archive_dmft.run[-1].system
    scc = run.m_create(Calculation)
    scc.system_ref = run.system[-1]
    template = add_template_dos(template)
    workflow = MaxEntworkflow()
    workflow_method = DMFTPlusMaxEntMethod(
        dmft_method_ref=archive_dmft.run[-1].method[-1].dmft,
        maxent_method_ref=archive_maxent.run[-1].method[-1],
    )
    workflow.m_add_sub_section(MaxEntworkflow.method, workflow_method)
    workflow.m_add_sub_section(
        MaxEntworkflow.inputs,
        Link(name="Input structure", section=archive_dmft.run[-1].system[-1]),
    )
    outputs = [
        Link(
            name="Output MaxEnt Sigma calculation",
            section=archive_dmft.run[-1].calculation[-1],
        ),
        Link(
            name="Output MaxEnt calculation", section=template.run[-1].calculation[-1]
        ),
    ]
    workflow.outputs = outputs
    workflow.m_add_sub_section(MaxEntworkflow.tasks, task_dmft)
    workflow.m_add_sub_section(MaxEntworkflow.tasks, task_maxent)
    template.workflow2 = workflow
    return template


def get_template_bse_workflow() -> EntryArchive:
    """Returns a basic archive template for a BSE workflow entry, composed of two tasks:
    PhotonPolarization SinglePoint number 1 and PhotonPolarization SinglePoint number 2."""
    # Adding two spectras for both photon polarizations
    archive_photon_1 = get_template_excited(type="Photon")
    archive_photon_2 = get_template_excited(type="Photon")
    n_energies = 11
    spectra_1 = Spectra(
        type="XAS",
        n_energies=n_energies,
        excitation_energies=np.linspace(0, 10, n_energies) * ureg.eV,
        intensities=np.linspace(100, 200, n_energies),
        intensities_units="F/m",
    )
    provenance_1 = ElectronicStructureProvenance(
        label="photon", methodology=archive_photon_1.run[-1].method[-1]
    )
    spectra_1.m_add_sub_section(Spectra.provenance, provenance_1)
    archive_photon_1.run[-1].calculation[-1].m_add_sub_section(
        Calculation.spectra, spectra_1
    )
    spectra_2 = Spectra(
        type="XAS",
        n_energies=n_energies,
        excitation_energies=np.linspace(0, 10, n_energies) * ureg.eV,
        intensities=np.linspace(200, 300, n_energies),
        intensities_units="F/m",
    )
    provenance_2 = ElectronicStructureProvenance(
        label="photon", methodology=archive_photon_2.run[-1].method[-1]
    )
    spectra_2.m_add_sub_section(Spectra.provenance, provenance_2)
    archive_photon_2.run[-1].calculation[-1].m_add_sub_section(
        Calculation.spectra, spectra_2
    )
    # Normalizing SinglePoint archives BEFORE defining the BSE workflow entry
    run_normalize(archive_photon_1)
    run_normalize(archive_photon_2)
    # Defining Photon1 and Photon2 tasks for later the BSE workflow
    task_photon_1 = TaskReference(task=archive_photon_1.workflow2)
    task_photon_1.name = "Photon 1"
    task_photon_1.inputs = [
        Link(name="Input structure", section=archive_photon_1.run[-1].system[-1])
    ]
    task_photon_1.outputs = [
        Link(
            name="Output polarization 1",
            section=archive_photon_1.run[-1].calculation[-1],
        )
    ]
    task_photon_2 = TaskReference(task=archive_photon_2.workflow2)
    task_photon_2.name = "Photon 2"
    task_photon_2.inputs = [
        Link(name="Input structure", section=archive_photon_1.run[-1].system[-1])
    ]
    task_photon_2.outputs = [
        Link(
            name="Output polarization 2",
            section=archive_photon_2.run[-1].calculation[-1],
        )
    ]
    # BSE workflow entry (no need of creating Calculation). We need to define BSE method.
    template = EntryArchive()
    run = template.m_create(Run)
    run.program = archive_photon_1.run[-1].program
    run.system = archive_photon_1.run[-1].system
    method = run.m_create(Method)
    method.bse = BSE(type="Singlet", solver="Lanczos-Haydock")
    workflow = PhotonPolarization()
    workflow.name = "BSE"
    workflow_method = PhotonPolarizationMethod(
        bse_method_ref=template.run[-1].method[-1].bse
    )
    workflow.m_add_sub_section(PhotonPolarization.method, workflow_method)
    spectras = [spectra_1, spectra_2]
    workflow_results = PhotonPolarizationResults(
        n_polarizations=2, spectrum_polarization=spectras
    )
    workflow.m_add_sub_section(PhotonPolarization.results, workflow_results)
    workflow.m_add_sub_section(
        PhotonPolarization.inputs,
        Link(name="Input structure", section=archive_photon_1.run[-1].system[-1]),
    )
    workflow.m_add_sub_section(
        PhotonPolarization.inputs,
        Link(name="Input BSE methodology", section=template.run[-1].method[-1]),
    )
    workflow.m_add_sub_section(
        PhotonPolarization.outputs,
        Link(
            name="Output polarization 1",
            section=archive_photon_1.run[-1].calculation[-1],
        ),
    )
    workflow.m_add_sub_section(
        PhotonPolarization.outputs,
        Link(
            name="Output polarization 2",
            section=archive_photon_2.run[-1].calculation[-1],
        ),
    )
    workflow.m_add_sub_section(PhotonPolarization.tasks, task_photon_1)
    workflow.m_add_sub_section(PhotonPolarization.tasks, task_photon_2)
    template.workflow2 = workflow
    return template


def get_template_xs_workflow() -> EntryArchive:
    """Returns a basic archive template for a XS workflow entry, composed of two main tasks:
    DFT GeometryOptimization and BSE workflow. The BSE workflow archive contains one
    PhotonPolarization SinglePoint task."""
    # Defining DFT and GW SinglePoint archives and adding band_structure and dos to them.
    archive_dft = get_template_dft()
    archive_dft = add_template_band_structure(archive_dft)
    archive_dft = add_template_dos(archive_dft)
    archive_bse = get_template_bse_workflow()
    # Normalizing SinglePoint archives BEFORE defining the XS workflow entry
    run_normalize(archive_dft)
    run_normalize(archive_bse)
    # Defining DFT and BSE tasks for later the BS workflow
    task_dft = TaskReference(task=archive_dft.workflow2)
    task_dft.name = "DFT"
    task_dft.inputs = [
        Link(name="Input structure", section=archive_dft.run[-1].system[-1])
    ]
    task_dft.outputs = [
        Link(name="Output DFT calculation", section=archive_dft.run[-1].calculation[-1])
    ]
    task_bse = TaskReference(task=archive_bse.workflow2)
    task_bse.name = "BSE 1"
    task_bse.inputs = [
        Link(name="Output DFT calculation", section=archive_dft.run[-1].calculation[-1])
    ]
    task_bse.outputs = [
        Link(name="Polarization 1", section=archive_bse.workflow2.outputs[0].section),
        Link(name="Polarization 2", section=archive_bse.workflow2.outputs[1].section),
    ]
    # XS (BSE) workflow entry (no need of creating Method nor Calculation)
    template = EntryArchive()
    run = template.m_create(Run)
    run.program = archive_dft.run[-1].program
    run.system = archive_dft.run[-1].system
    workflow = XSworkflow()
    workflow.name = "XS"
    workflow.m_add_sub_section(
        XSworkflow.inputs,
        Link(name="Input structure", section=archive_dft.run[-1].system[-1]),
    )
    workflow.m_add_sub_section(
        XSworkflow.outputs,
        Link(name="Polarization 1", section=archive_bse.workflow2.outputs[0].section),
    )
    workflow.m_add_sub_section(
        XSworkflow.outputs,
        Link(name="Polarization 2", section=archive_bse.workflow2.outputs[1].section),
    )
    workflow.m_add_sub_section(XSworkflow.tasks, task_dft)
    workflow.m_add_sub_section(XSworkflow.tasks, task_bse)
    template.workflow2 = workflow
    return template


def set_dft_values(xc_functional_names: list) -> EntryArchive:
    """"""
    template = get_template_dft()
    template.run[0].method = None
    run = template.run[0]
    method_dft = run.m_create(Method)
    method_dft.electrons_representation = [
        BasisSetContainer(
            type="plane waves",
            scope=["wavefunction"],
            basis_set=[
                BasisSet(
                    type="plane waves",
                    scope=["valence"],
                    cutoff=500,
                )
            ],
        )
    ]
    method_dft.dft = DFT()
    method_dft.electronic = Electronic(
        method="DFT",
        smearing=Smearing(kind="gaussian", width=1e-20),
        n_spin_channels=2,
        van_der_waals_method="G06",
        relativity_method="scalar_relativistic",
    )
    method_dft.scf = Scf(threshold_energy_change=1e-24)
    method_dft.dft.xc_functional = XCFunctional()
    xc = method_dft.dft.xc_functional
    for xc_functional_name in xc_functional_names:
        if re.search("^HYB_", xc_functional_name):
            xc.hybrid.append(Functional(name=xc_functional_name, weight=1.0))
            continue
        if re.search("_X?C_", xc_functional_name):
            xc.correlation.append(Functional(name=xc_functional_name, weight=1.0))
        if re.search("_XC?_", xc_functional_name):
            xc.exchange.append(Functional(name=xc_functional_name, weight=1.0))
        xc.correlation.append(Functional(name=xc_functional_name, weight=1.0))
    return template


@pytest.fixture(scope="session")
def dft() -> EntryArchive:
    """DFT calculation."""
    template = set_dft_values(["GGA_C_PBE", "GGA_X_PBE"])
    return run_normalize(template)


@pytest.fixture(scope="session")
def dft_method_referenced() -> EntryArchive:
    """DFT calculation with two methods: one referencing the other."""
    template = get_template_dft()
    template.run[0].method = None
    run = template.run[0]
    method_dft = run.m_create(Method)
    method_dft.electrons_representation = [
        BasisSetContainer(
            type="plane waves",
            scope=["wavefunction"],
            basis_set=[
                BasisSet(
                    type="plane waves",
                    scope=["valence"],
                )
            ],
        )
    ]
    method_dft.electronic = Electronic(
        smearing=Smearing(kind="gaussian", width=1e-20),
        n_spin_channels=2,
        van_der_waals_method="G06",
        relativity_method="scalar_relativistic",
    )
    method_dft.scf = Scf(threshold_energy_change=1e-24)
    method_dft.dft = DFT(xc_functional=XCFunctional())
    method_dft.dft.xc_functional.correlation.append(
        Functional(name="GGA_C_PBE", weight=1.0)
    )
    method_dft.dft.xc_functional.exchange.append(
        Functional(name="GGA_X_PBE", weight=1.0)
    )
    method_ref = run.m_create(Method)
    method_ref.electrons_representation = [
        BasisSetContainer(
            type="plane waves",
            scope=["wavefunction"],
            basis_set=[
                BasisSet(
                    type="plane waves",
                    scope=["valence"],
                )
            ],
        )
    ]
    method_ref.electronic = Electronic(method="DFT")
    method_ref.core_method_ref = method_dft
    run.calculation[0].method_ref = method_ref
    return run_normalize(template)


@pytest.fixture(scope="session")
def dft_exact_exchange() -> EntryArchive:
    """Add exact exchange explicitely to a PBE calculation."""
    template = set_dft_values(["GGA_C_PBE", "GGA_X_PBE"])
    template.run[0].method[0].dft.xc_functional.hybrid.append(Functional())
    template.run[0].method[0].dft.xc_functional.hybrid[0].parameters = {
        "exact_exchange_mixing_factor": 0.25
    }
    template.run[0].method[0].dft.xc_functional.hybrid[0].name = "+alpha"
    return run_normalize(template)


@pytest.fixture(scope="session")
def dft_empty() -> EntryArchive:
    template = set_dft_values([])
    return run_normalize(template)


@pytest.fixture(scope="session")
def dft_wrong() -> EntryArchive:
    template = set_dft_values(["FOO_X_FIGHTERS", "BAR_C_EXAM"])
    return run_normalize(template)


@pytest.fixture(scope="session")
def dft_pw() -> EntryArchive:
    template = set_dft_values(["LDA_X", "LDA_C_PW"])
    return run_normalize(template)


@pytest.fixture(scope="session")
def dft_m06() -> EntryArchive:
    template = set_dft_values(["MGGA_X_M06", "MGGA_C_M06"])
    return run_normalize(template)


@pytest.fixture(scope="session")
def dft_b3lyp() -> EntryArchive:
    template = set_dft_values(["HYB_GGA_XC_B3LYP"])
    return run_normalize(template)


@pytest.fixture(scope="session")
def dft_pbeh() -> EntryArchive:
    template = set_dft_values(["HYB_GGA_XC_PBEH"])
    return run_normalize(template)


@pytest.fixture(scope="session")
def dft_m05() -> EntryArchive:
    template = set_dft_values(["MGGA_C_M05", "HYB_MGGA_X_M05"])
    return run_normalize(template)


@pytest.fixture(scope="session")
def dft_pbe0_13() -> EntryArchive:
    template = set_dft_values(["HYB_GGA_XC_PBE0_13"])
    return run_normalize(template)


@pytest.fixture(scope="session")
def dft_pbe38() -> EntryArchive:
    template = set_dft_values(["HYB_GGA_XC_PBE38"])
    return run_normalize(template)


@pytest.fixture(scope="session")
def dft_pbe50() -> EntryArchive:
    template = set_dft_values(["HYB_GGA_XC_PBE50"])
    return run_normalize(template)


@pytest.fixture(scope="session")
def dft_m06_2x() -> EntryArchive:
    template = set_dft_values(["HYB_MGGA_X_M06_2X"])
    return run_normalize(template)


@pytest.fixture(scope="session")
def dft_m05_2x() -> EntryArchive:
    template = set_dft_values(["MGGA_C_M05_2X", "HYB_MGGA_X_M05_2X"])
    return run_normalize(template)


@pytest.fixture(scope="session")
def dft_plus_u() -> EntryArchive:
    """DFT+U calculation with a Hubbard model."""
    template = get_template_dft()
    template.run[0].method = None
    run = template.run[0]
    method_dft = run.m_create(Method)
    method_dft.electrons_representation = [
        BasisSetContainer(
            type="plane waves",
            scope=["wavefunction"],
            basis_set=[
                BasisSet(
                    type="plane waves",
                    scope=["valence"],
                )
            ],
        )
    ]
    method_dft.electronic = Electronic(
        method="DFT+U",
        smearing=Smearing(kind="gaussian", width=1e-20),
        n_spin_channels=2,
        van_der_waals_method="G06",
        relativity_method="scalar_relativistic",
    )
    method_dft.scf = Scf(threshold_energy_change=1e-24)
    method_dft.dft = DFT(xc_functional=XCFunctional())
    method_dft.dft.xc_functional.correlation.append(
        Functional(name="GGA_C_PBE", weight=1.0)
    )
    method_dft.dft.xc_functional.exchange.append(
        Functional(name="GGA_X_PBE", weight=1.0)
    )
    method_dft.atom_parameters.append(AtomParameters(label="Ti"))
    method_dft.atom_parameters[0].hubbard_kanamori_model = HubbardKanamoriModel(
        orbital="3d", u=4.5e-19, j=1.0e-19, double_counting_correction="Dudarev"
    )
    return run_normalize(template)


@pytest.fixture(scope="session")
def tb_wannier() -> EntryArchive:
    """Wannier TB calculation."""
    template = get_template_tb_wannier()
    return run_normalize(template)


@pytest.fixture(scope="session")
def gw() -> EntryArchive:
    """GW calculation."""
    template = get_template_excited(type="GW")
    return run_normalize(template)


@pytest.fixture(scope="session")
def bse() -> EntryArchive:
    """BSE calculation."""
    template = get_template_excited(type="BSE")
    return run_normalize(template)


@pytest.fixture(scope="session")
def dmft() -> EntryArchive:
    """DMFT calculation."""
    template = get_template_dmft()
    return run_normalize(template)


@pytest.fixture(scope="session")
def mechanical_elastic() -> EntryArchive:
    """Entry with mechanical properties."""
    template = get_template_dft()

    # Elastic workflow
    template.workflow2 = Elastic()
    template.workflow2.results = ElasticResults(
        shear_modulus_hill=10000,
        shear_modulus_reuss=10000,
        shear_modulus_voigt=10000,
    )

    return run_normalize(template)


@pytest.fixture(scope="session")
def mechanical_eos() -> EntryArchive:
    """Entry with mechanical properties."""
    template = get_template_dft()

    # EOS workflow
    template.workflow2 = EquationOfState()
    template.workflow2.results = EquationOfStateResults(
        volumes=np.linspace(0, 10, 10) * ureg.angstrom**3,
        energies=np.linspace(0, 10, 10) * ureg.electron_volt,
    )
    eos_fit = template.workflow2.results.m_create(EOSFit)
    eos_fit.function_name = "murnaghan"
    eos_fit.fitted_energies = np.linspace(0, 10, 10) * ureg.electron_volt
    eos_fit.bulk_modulus = 10000

    return run_normalize(template)


@pytest.fixture(scope="session")
def single_point() -> EntryArchive:
    """Single point calculation."""
    template = get_template_dft()
    return run_normalize(template)


@pytest.fixture(scope="session")
def gw_workflow() -> EntryArchive:
    """GW workflow (DFT+GW) EntryArchive."""
    template = get_template_gw_workflow()
    return run_normalize(template)


@pytest.fixture(scope="session")
def dmft_workflow() -> EntryArchive:
    """DMFT workflow (Projection+DMFT) EntryArchive."""
    template = get_template_dmft_workflow()
    return run_normalize(template)


@pytest.fixture(scope="session")
def maxent_workflow() -> EntryArchive:
    """MaxEnt workflow (DMFT+MaxEnt Sigma) EntryArchive."""
    template = get_template_maxent_workflow()
    return run_normalize(template)


@pytest.fixture(scope="session")
def bse_workflow() -> EntryArchive:
    """BSE workflow (Photon1+Photon2) EntryArchive"""
    template = get_template_bse_workflow()
    return run_normalize(template)


@pytest.fixture(scope="session")
def xs_workflow() -> EntryArchive:
    """XS workflow (DFT+BSEworkflow) EntryArchive."""
    template = get_template_xs_workflow()
    return run_normalize(template)


@pytest.fixture(scope="session")
def geometry_optimization() -> EntryArchive:
    template = get_template_dft()
    template.run[0].system = None
    template.run[0].calculation = None
    run = template.run[0]
    atoms1 = ase.build.bulk("Si", "diamond", cubic=True, a=5.431)
    atoms2 = ase.build.bulk("Si", "diamond", cubic=True, a=5.431)
    atoms2.translate([0.01, 0, 0])
    sys1 = get_section_system(atoms1)
    sys2 = get_section_system(atoms2)
    scc1 = run.m_create(Calculation)
    scc2 = run.m_create(Calculation)
    scc1.energy = Energy(
        total=EnergyEntry(value=1e-19), total_t0=EnergyEntry(value=1e-19)
    )
    scc2.energy = Energy(
        total=EnergyEntry(value=0.5e-19), total_t0=EnergyEntry(value=0.5e-19)
    )
    scc1.system_ref = sys1
    scc2.system_ref = sys2
    scc1.method_ref = run.method[0]
    scc2.method_ref = run.method[0]
    run.m_add_sub_section(Run.system, sys1)
    run.m_add_sub_section(Run.system, sys2)
    template.workflow2 = GeometryOptimization(
        method=GeometryOptimizationMethod(
            convergence_tolerance_energy_difference=1e-3 * ureg.electron_volt,
            convergence_tolerance_force_maximum=1e-11 * ureg.newton,
            convergence_tolerance_displacement_maximum=1e-3 * ureg.angstrom,
            method="bfgs",
        )
    )
    return run_normalize(template)


@pytest.fixture(scope="session")
def molecular_dynamics() -> EntryArchive:
    """Molecular dynamics calculation."""
    template = get_template_dft()
    run = template.run[0]

    # Create calculations
    n_steps = 10
    calcs = []
    for step in range(n_steps):
        system = System()
        run.m_add_sub_section(Run.system, system)
        calc = Calculation()
        calc.system_ref = system
        calc.time = step
        calc.step = step
        calc.volume = step
        calc.pressure = step
        calc.temperature = step
        calc.energy = Energy(
            potential=EnergyEntry(value=step),
        )
        rg_values = RadiusOfGyrationValues(
            value=step, label="MOL", atomsgroup_ref=system
        )
        calc.radius_of_gyration = [
            RadiusOfGyration(
                kind="molecular",
                radius_of_gyration_values=[rg_values],
            )
        ]
        calcs.append(calc)
        run.m_add_sub_section(Run.calculation, calc)

    # Create workflow
    diff_values = DiffusionConstantValues(
        value=2.1,
        error_type="Pearson correlation coefficient",
        errors=0.98,
    )
    msd_values = MeanSquaredDisplacementValues(
        times=[0, 1, 2],
        n_times=3,
        value=[0, 1, 2],
        label="MOL",
        errors=[0, 1, 2],
        diffusion_constant=diff_values,
    )
    msd = MeanSquaredDisplacement(
        type="molecular",
        direction="xyz",
        error_type="bootstrapping",
        mean_squared_displacement_values=[msd_values],
    )
    rdf_values = RadialDistributionFunctionValues(
        bins=[0, 1, 2],
        n_bins=3,
        value=[0, 1, 2],
        frame_start=0,
        frame_end=100,
        label="MOL-MOL",
    )
    rdf = RadialDistributionFunction(
        type="molecular",
        radial_distribution_function_values=[rdf_values],
    )
    results = MolecularDynamicsResults(
        radial_distribution_functions=[rdf],
        mean_squared_displacements=[msd],
    )
    method = MolecularDynamicsMethod(
        thermodynamic_ensemble="NVT",
        integration_timestep=0.5 * ureg("fs"),
    )
    md = MolecularDynamics(results=results, method=method)
    results.calculation_result_ref = calcs[-1]
    results.calculations_ref = calcs
    template.workflow2 = md

    return run_normalize(template)


def get_template_topology(pbc=False) -> EntryArchive:
    template = get_template_dft()
    run = template.run[0]
    del run.system[0]

    # System
    water1 = ase.build.molecule("H2O")
    water2 = ase.build.molecule("H2O")
    water2.translate([5, 0, 0])
    sys = water1 + water2
    sys.set_cell([10, 10, 10])
    sys.set_pbc(pbc)
    system = get_section_system(sys)
    run.m_add_sub_section(Run.system, system)

    # Topology
    molecule_group = AtomsGroup(
        label="MOL_GROUP",
        type="molecule_group",
        index=0,
        composition_formula="H(4)O(2)",
        n_atoms=6,
        atom_indices=[0, 1, 2, 3, 4, 5],
    )
    system.m_add_sub_section(System.atoms_group, molecule_group)
    molecule1 = AtomsGroup(
        label="MOL",
        type="molecule",
        index=0,
        composition_formula="H(2)O(1)",
        n_atoms=3,
        atom_indices=[0, 1, 2],
    )
    molecule_group.m_add_sub_section(AtomsGroup.atoms_group, molecule1)
    molecule2 = AtomsGroup(
        label="MOL",
        type="molecule",
        index=0,
        composition_formula="H(2)O(1)",
        n_atoms=3,
        atom_indices=[3, 4, 5],
    )
    molecule_group.m_add_sub_section(AtomsGroup.atoms_group, molecule2)
    monomer_group = AtomsGroup(
        label="MON_GROUP",
        type="monomer_group",
        index=0,
        composition_formula="H(2)",
        n_atoms=2,
        atom_indices=[1, 2],
    )
    molecule1.m_add_sub_section(AtomsGroup.atoms_group, monomer_group)
    monomer = AtomsGroup(
        label="MON",
        type="monomer",
        index=0,
        composition_formula="H(2)",
        n_atoms=2,
        atom_indices=[1, 2],
    )
    monomer_group.m_add_sub_section(AtomsGroup.atoms_group, monomer)

    # Calculation
    calc = Calculation()
    calc.system_ref = system
    run.m_add_sub_section(Run.calculation, calc)

    return run_normalize(template)
