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

import pytest
from ase.io import Trajectory
import random
import numpy as np
import os

from nomad.units import ureg
from nomad.utils import get_logger
from nomad.datamodel import EntryArchive, EntryMetadata
from nomad.datamodel.metainfo.simulation.run import Run, Program
from nomad.datamodel.metainfo.simulation.calculation import (
    Calculation,
    Energy,
    EnergyEntry,
)
from nomad.datamodel.metainfo.simulation.system import System, Atoms
from nomad.datamodel.metainfo.workflow import Task, Link
from simulationworkflowschema.general import (
    SimulationWorkflow,
    SimulationWorkflowMethod,
    SimulationWorkflowResults,
)
from simulationworkflowschema.equation_of_state import EquationOfState
from simulationworkflowschema.chemical_reaction import ChemicalReaction


from electronicparsers.vasp import VASPParser
from atomisticparsers.gromacs import GromacsParser
from atomisticparsers.lammps import LammpsParser
from workflowparsers.elastic import ElasticParser
from workflowparsers.phonopy import PhonopyParser
from conftest import run_normalize


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


LOGGER = get_logger(__name__)


def run_parsing(parser_class, filepath):
    archive = EntryArchive(metadata=EntryMetadata())
    parser_class().parse(filepath, archive, LOGGER)
    run_normalize(archive)
    return archive


# TODO rid of the use of parser


def test_gw_workflow(gw_workflow):
    """Testing GW workflow (DFT+GW) entry"""
    workflow = gw_workflow.workflow2
    assert workflow.name == "DFT+GW"
    assert workflow.method.gw_method_ref.type == "G0W0"
    assert workflow.method.electrons_representation.type == "plane waves"
    assert workflow.method.starting_point.name == "GGA_X_PBE"
    results = gw_workflow.results
    assert results.method.method_name == "GW"
    assert results.method.workflow_name == "DFT+GW"
    assert results.method.simulation.program_name == "VASP"
    assert results.method.simulation.program_version == "4.6.35"
    assert results.method.simulation.gw.type == "G0W0"
    assert results.method.simulation.gw.starting_point_type == "GGA"
    assert results.method.simulation.gw.starting_point_names == ["GGA_X_PBE"]
    assert results.method.simulation.gw.basis_set_type == "plane waves"
    assert not results.properties.electronic.band_gap
    assert not results.properties.electronic.greens_functions_electronic
    assert len(results.properties.electronic.dos_electronic_new) == 2
    assert len(results.properties.electronic.band_structure_electronic) == 2
    assert results.properties.electronic.dos_electronic_new[0].label == "DFT"
    assert results.properties.electronic.dos_electronic_new[1].label == "GW"


def test_dmft_workflow(dmft_workflow):
    """Testing DMFT workflow entry"""
    workflow = dmft_workflow.workflow2
    assert workflow.name == "DFT+TB+DMFT"
    assert not workflow.method.tb_method_ref.wannier.is_maximally_localized
    assert workflow.method.dmft_method_ref.n_impurities == 1
    assert workflow.method.dmft_method_ref.n_correlated_orbitals[0] == 3
    assert workflow.method.dmft_method_ref.n_electrons[0] == 1.0
    assert workflow.method.dmft_method_ref.inverse_temperature.magnitude == 60.0
    assert workflow.method.dmft_method_ref.magnetic_state == "paramagnetic"
    assert workflow.method.dmft_method_ref.impurity_solver == "CT-HYB"
    results = dmft_workflow.results
    assert results.method.method_name == "DMFT"
    assert results.method.workflow_name == "DFT+TB+DMFT"
    assert results.method.simulation.program_name == "w2dynamics"
    assert results.method.simulation.dmft.impurity_solver_type == "CT-HYB"
    assert results.method.simulation.dmft.inverse_temperature.magnitude == 60.0
    assert results.method.simulation.dmft.magnetic_state == "paramagnetic"
    assert results.method.simulation.dmft.u.magnitude == 4.0e-19
    assert results.method.simulation.dmft.jh.magnitude == 0.6e-19
    assert results.m_xpath("properties.electronic.band_gap")
    assert len(results.properties.electronic.band_gap) == 1
    assert results.properties.electronic.band_gap[0].label == "TB"
    assert results.m_xpath("properties.electronic.band_structure_electronic")
    assert len(results.properties.electronic.band_structure_electronic) == 1
    # TODO check why this testing is not passing
    #   * conftest seems to not be able to normalize the archive_dmft for the Greens functions, despite self_energy_iw is defined.
    # assert results.m_xpath('properties.electronic.greens_function_electronic')


def test_maxent_workflow(maxent_workflow):
    """Testing MaxEnt workflow entry"""
    workflow = maxent_workflow.workflow2
    assert workflow.name == "DMFT+MaxEnt"
    assert workflow.method.dmft_method_ref.n_impurities == 1
    assert workflow.method.dmft_method_ref.n_correlated_orbitals[0] == 3
    assert workflow.method.dmft_method_ref.n_electrons[0] == 1.0
    assert workflow.method.dmft_method_ref.inverse_temperature.magnitude == 60.0
    assert workflow.method.dmft_method_ref.magnetic_state == "paramagnetic"
    assert workflow.method.dmft_method_ref.impurity_solver == "CT-HYB"
    assert workflow.method.maxent_method_ref
    results = maxent_workflow.results
    assert results.method.method_name == "DMFT"
    assert results.method.workflow_name == "DMFT+MaxEnt"
    assert results.method.simulation.program_name == "w2dynamics"
    assert results.method.simulation.dmft.impurity_solver_type == "CT-HYB"
    assert results.method.simulation.dmft.inverse_temperature.magnitude == 60.0
    assert results.method.simulation.dmft.magnetic_state == "paramagnetic"
    assert results.method.simulation.dmft.u.magnitude == 4.0e-19
    assert results.method.simulation.dmft.jh.magnitude == 0.6e-19
    assert results.method.simulation.dmft.analytical_continuation == "MaxEnt"
    assert results.m_xpath("properties.electronic.dos_electronic_new")
    assert len(results.properties.electronic.dos_electronic_new) == 1
    assert results.m_xpath("properties.electronic.greens_functions_electronic")
    assert len(results.properties.electronic.greens_functions_electronic) == 2
    assert results.properties.electronic.greens_functions_electronic[0].label == "DMFT"
    assert (
        results.properties.electronic.greens_functions_electronic[1].label == "MaxEnt"
    )


def test_bse_workflow(bse_workflow):
    """Testing BSE workflow (Photon1+Photon2) entry"""
    workflow = bse_workflow.workflow2
    assert workflow.name == "BSE"
    assert len(workflow.inputs) == 2
    assert workflow.inputs[0].name == "Input structure"
    assert workflow.inputs[1].name == "Input BSE methodology"
    assert (
        len(workflow.outputs) == 2
        and len(workflow.outputs) == workflow.results.n_polarizations
    )
    assert len(workflow.tasks) == 2
    assert workflow.method.bse_method_ref.type == "Singlet"
    assert workflow.method.bse_method_ref.solver == "Lanczos-Haydock"
    results = bse_workflow.results
    assert results.method.method_name == "BSE"
    assert results.method.workflow_name == "BSE"
    assert results.method.simulation.program_name == "VASP"
    assert results.method.simulation.program_version == "4.6.35"
    assert results.method.simulation.bse.type == "Singlet"
    assert results.method.simulation.bse.solver == "Lanczos-Haydock"
    assert results.properties.spectroscopic
    spectra = results.properties.spectroscopic.spectra
    assert len(spectra) == 2
    assert spectra[0].type == "XAS"
    assert spectra[0].label == "computation"
    assert spectra[0].n_energies == 11
    assert spectra[0].energies[3].to("eV").magnitude == approx(3.0)
    assert spectra[0].intensities[3] == approx(130.0)
    assert spectra[0].intensities_units == "F/m"
    assert spectra[0].provenance and spectra[1].provenance
    assert spectra[0].provenance != spectra[1].provenance


def test_xs_workflow(xs_workflow):
    """Testing XS workflow (DFT+BSEworkflow) entry"""
    workflow = xs_workflow.workflow2
    assert workflow.name == "XS"
    assert len(workflow.inputs) == 1
    assert workflow.inputs[0].name == "Input structure"
    assert len(workflow.outputs) == 2
    assert len(workflow.tasks) == 2
    assert workflow.tasks[0].name == "DFT" and workflow.tasks[1].name == "BSE 1"
    assert (
        workflow.results.dft_outputs.dos
        and workflow.results.dft_outputs.band_structure
        and workflow.results.spectra
    )
    results = xs_workflow.results
    assert results.method.method_name == "BSE"
    assert results.method.workflow_name == "XS"
    assert results.method.simulation.program_name == "VASP"
    assert results.method.simulation.program_version == "4.6.35"
    assert results.method.simulation.bse.type == "Singlet"
    assert results.method.simulation.bse.solver == "Lanczos-Haydock"
    assert results.method.simulation.bse.starting_point_type == "GGA"
    assert results.method.simulation.bse.starting_point_names == ["GGA_X_PBE"]
    assert results.method.simulation.bse.basis_set_type == "plane waves"
    assert results.properties.electronic and results.properties.spectroscopic
    assert results.properties.electronic.dos_electronic_new[0].label == "DFT"
    assert len(results.properties.spectroscopic.spectra) == 2
    assert (
        results.properties.spectroscopic.spectra[0].provenance
        != results.properties.spectroscopic.spectra[1].provenance
    )


def test_geometry_optimization_workflow():
    vasp_archive = run_parsing(VASPParser, "tests/data/vasp/vasprun.xml")
    sec_workflow = vasp_archive.workflow2
    assert sec_workflow.method.type == "cell_shape"
    assert sec_workflow.results.calculation_result_ref.m_def.name == "Calculation"
    assert sec_workflow.results.final_energy_difference.to("eV").magnitude == approx(
        0.00012532
    )
    assert sec_workflow.results.optimization_steps == 3
    assert sec_workflow.results.final_force_maximum > 0.0
    assert sec_workflow.results.is_converged_geometry

    tasks = sec_workflow.tasks
    assert len(tasks) == len(vasp_archive.run[0].calculation)
    assert tasks[0].inputs[0].section == vasp_archive.run[0].method[0]
    assert tasks[-1].outputs[0].section == vasp_archive.run[0].calculation[-1]


def test_elastic_workflow():
    elastic_archive = run_parsing(ElasticParser, "tests/data/elastic/INFO_ElaStic")
    sec_workflow = elastic_archive.workflow2
    sec_workflow.results.calculation_result_ref.m_def.name == "Calculation"
    sec_workflow.method.calculation_method == "energy"
    sec_workflow.method.elastic_constants_order == 2
    sec_workflow.results.is_mechanically_stable
    sec_workflow.method.fitting_error_maximum > 0.0
    sec_workflow.method.strain_maximum > 0.0


def test_phonon_workflow():
    phonopy_archive = run_parsing(
        PhonopyParser, "tests/data/phonopy/phonopy-FHI-aims-displacement-01/control.in"
    )

    sec_workflow = phonopy_archive.workflow2
    assert sec_workflow.results.calculation_result_ref.m_def.name == "Calculation"
    assert sec_workflow.method.force_calculator == "fhi-aims"
    assert sec_workflow.method.mesh_density > 0.0
    assert sec_workflow.results.n_imaginary_frequencies > 0
    assert not sec_workflow.method.random_displacements
    assert not sec_workflow.method.with_non_analytic_correction
    assert not sec_workflow.method.with_grueneisen_parameters


def test_molecular_dynamics_workflow():
    lammmps_archive = run_parsing(
        LammpsParser, "tests/data/lammps/64xmethane/log.lammps"
    )

    sec_workflow = lammmps_archive.workflow2
    sec_workflow.results.calculation_result_ref.m_def.name == "Calculation"
    assert sec_workflow.results.finished_normally
    assert sec_workflow.results.trajectory


def test_rdf_and_msd():
    archive = run_parsing(
        LammpsParser, "tests/data/lammps/hexane_cyclohexane/log.hexane_cyclohexane_nvt"
    )

    sec_workflow = archive.workflow2
    section_md = sec_workflow.results

    assert section_md.radial_distribution_functions[0].type == "molecular"
    assert section_md.radial_distribution_functions[0].n_smooth == 2
    assert section_md.radial_distribution_functions[0].variables_name[0] == "distance"

    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[0]
        .label
        == "0-0"
    )
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[0]
        .n_bins
        == 198
    )
    assert section_md.radial_distribution_functions[
        0
    ].radial_distribution_function_values[0].bins[122].magnitude == approx(
        6.923255643844605 * 10 ** (-10)
    )
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[0]
        .bins[122]
        .units
        == "meter"
    )
    assert section_md.radial_distribution_functions[
        0
    ].radial_distribution_function_values[0].value[96] == approx(0.0)
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[0]
        .frame_start
        == 0
    )
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[0]
        .frame_end
        == 40
    )

    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[3]
        .label
        == "0-0"
    )
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[3]
        .n_bins
        == 198
    )
    assert section_md.radial_distribution_functions[
        0
    ].radial_distribution_function_values[3].bins[65].magnitude == approx(
        3.727906885147095 * 10 ** (-10)
    )
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[3]
        .bins[65]
        .units
        == "meter"
    )
    assert section_md.radial_distribution_functions[
        0
    ].radial_distribution_function_values[3].value[52] == approx(0.0)
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[3]
        .frame_start
        == 120
    )
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[3]
        .frame_end
        == 201
    )

    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[5]
        .label
        == "1-0"
    )
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[5]
        .n_bins
        == 198
    )
    assert section_md.radial_distribution_functions[
        0
    ].radial_distribution_function_values[5].bins[102].magnitude == approx(
        5.802080640792847 * 10 ** (-10)
    )
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[5]
        .bins[102]
        .units
        == "meter"
    )
    assert section_md.radial_distribution_functions[
        0
    ].radial_distribution_function_values[5].value[55] == approx(0.0)
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[5]
        .frame_start
        == 40
    )
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[5]
        .frame_end
        == 201
    )

    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[10]
        .label
        == "1-1"
    )
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[10]
        .n_bins
        == 198
    )
    assert section_md.radial_distribution_functions[
        0
    ].radial_distribution_function_values[10].bins[44].magnitude == approx(
        2.550673131942749 * 10 ** (-10)
    )
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[10]
        .bins[44]
        .units
        == "meter"
    )
    assert section_md.radial_distribution_functions[
        0
    ].radial_distribution_function_values[10].value[101] == approx(1.4750986777470825)
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[10]
        .frame_start
        == 80
    )
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[10]
        .frame_end
        == 201
    )

    assert section_md.mean_squared_displacements[0].type == "molecular"
    assert section_md.mean_squared_displacements[0].direction == "xyz"

    assert (
        section_md.mean_squared_displacements[0]
        .mean_squared_displacement_values[0]
        .label
        == "0"
    )
    assert (
        section_md.mean_squared_displacements[0]
        .mean_squared_displacement_values[0]
        .n_times
        == 54
    )
    assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[
        0
    ].times[13].magnitude == approx(1.3 * 10 ** (-12))
    assert (
        section_md.mean_squared_displacements[0]
        .mean_squared_displacement_values[0]
        .times[13]
        .units
        == "second"
    )
    assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[
        0
    ].value[32].magnitude == approx(8.98473539965496 * 10 ** (-19))
    assert (
        section_md.mean_squared_displacements[0]
        .mean_squared_displacement_values[0]
        .value[32]
        .units
        == "meter^2"
    )
    assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[
        0
    ].diffusion_constant.value.magnitude == approx(6.09812270414572 * 10 ** (-8))
    assert (
        section_md.mean_squared_displacements[0]
        .mean_squared_displacement_values[0]
        .diffusion_constant.value.units
        == "meter^2/second"
    )
    assert (
        section_md.mean_squared_displacements[0]
        .mean_squared_displacement_values[0]
        .diffusion_constant.error_type
        == "Pearson correlation coefficient"
    )
    assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[
        0
    ].diffusion_constant.errors == approx(0.9924847048341159)

    assert (
        section_md.mean_squared_displacements[0]
        .mean_squared_displacement_values[1]
        .label
        == "1"
    )
    assert (
        section_md.mean_squared_displacements[0]
        .mean_squared_displacement_values[1]
        .n_times
        == 54
    )
    assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[
        1
    ].times[13].magnitude == approx(1.3 * 10 ** (-12))
    assert (
        section_md.mean_squared_displacements[0]
        .mean_squared_displacement_values[1]
        .times[13]
        .units
        == "second"
    )
    assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[
        1
    ].value[32].magnitude == approx(8.448369705677565 * 10 ** (-19))
    assert (
        section_md.mean_squared_displacements[0]
        .mean_squared_displacement_values[1]
        .value[32]
        .units
        == "meter^2"
    )
    assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[
        1
    ].diffusion_constant.value.magnitude == approx(5.094072039759048 * 10 ** (-8))
    assert (
        section_md.mean_squared_displacements[0]
        .mean_squared_displacement_values[1]
        .diffusion_constant.value.units
        == "meter^2/second"
    )
    assert (
        section_md.mean_squared_displacements[0]
        .mean_squared_displacement_values[1]
        .diffusion_constant.error_type
        == "Pearson correlation coefficient"
    )
    assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[
        1
    ].diffusion_constant.errors == approx(0.9965870174917716)


def test_rdf_2():
    archive = run_parsing(GromacsParser, "tests/data/gromacs/fe_test/mdrun.out")

    sec_workflow = archive.workflow2
    section_md = sec_workflow.results

    assert section_md.radial_distribution_functions[0].type == "molecular"
    assert section_md.radial_distribution_functions[0].n_smooth == 2
    assert section_md.radial_distribution_functions[0].variables_name[0] == "distance"

    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[0]
        .label
        == "SOL-Protein"
    )
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[0]
        .n_bins
        == 198
    )
    assert section_md.radial_distribution_functions[
        0
    ].radial_distribution_function_values[0].bins[122].magnitude == approx(
        7.624056451320648 * 10 ** (-10)
    )
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[0]
        .bins[122]
        .units
        == "meter"
    )
    assert section_md.radial_distribution_functions[
        0
    ].radial_distribution_function_values[0].value[96] == approx(1.093694948374587)
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[0]
        .frame_start
        == 0
    )
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[0]
        .frame_end
        == 2
    )

    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[1]
        .label
        == "SOL-SOL"
    )
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[1]
        .n_bins
        == 198
    )
    assert section_md.radial_distribution_functions[
        0
    ].radial_distribution_function_values[1].bins[102].magnitude == approx(
        6.389391438961029 * 10 ** (-10)
    )
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[1]
        .bins[102]
        .units
        == "meter"
    )
    assert section_md.radial_distribution_functions[
        0
    ].radial_distribution_function_values[1].value[55] == approx(0.8368052672121375)
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[1]
        .frame_start
        == 0
    )
    assert (
        section_md.radial_distribution_functions[0]
        .radial_distribution_function_values[1]
        .frame_end
        == 2
    )


def test_radius_of_gyration():
    archive = run_parsing(
        GromacsParser, "tests/data/gromacs/protein_small_nowater/md.log"
    )

    sec_calc = archive.run[0].calculation[4]
    sec_rg = sec_calc.radius_of_gyration[0]
    sec_rgvals = sec_rg.radius_of_gyration_values[0]

    assert sec_rg.kind == "molecular"

    assert sec_rgvals.label == "Protein_chain_X-index_0"
    assert sec_rgvals.value.magnitude == approx(5.081165959952965e-10)
    assert sec_rgvals.value.units == "meter"

    sec_calc = archive.run[0].calculation[1]
    sec_rg = sec_calc.radius_of_gyration[0]
    sec_rgvals = sec_rg.radius_of_gyration_values[0]

    assert sec_rg.kind == "molecular"
    assert sec_rgvals.label == "Protein_chain_X-index_0"
    assert sec_rgvals.value.magnitude == approx(5.036762961380965e-10)
    assert sec_rgvals.value.units == "meter"

    sec_workflow = archive.workflow2
    sec_rg = sec_workflow.results.radius_of_gyration[0]
    frame = 4

    assert sec_rg.type == "molecular"

    assert sec_rg.label == "Protein_chain_X-index_0"
    assert sec_rg.value[frame].magnitude == approx(5.081165959952965e-10)
    assert sec_rg.value[frame].units == "meter"

    frame = 1
    sec_rg = sec_workflow.results.radius_of_gyration[0]
    sec_calc = archive.run[0].calculation[1]

    assert sec_rg.type == "molecular"
    assert sec_rg.label == "Protein_chain_X-index_0"
    assert sec_rg.value[frame].magnitude == approx(5.036762961380965e-10)
    assert sec_rg.value[frame].units == "meter"


def parse_trajectory(filename):
    # TODO implement parser for ase trajectory
    trajectory = Trajectory(filename)

    archive = EntryArchive()

    run = Run(program=Program(name="ASE"))
    for frame in trajectory:
        calc = Calculation(
            energy=Energy(
                total=EnergyEntry(value=frame.get_potential_energy() * ureg.eV)
            )
        )
        system = System(
            atoms=Atoms(
                positions=frame.get_positions() * ureg.angstrom,
                labels=frame.get_chemical_symbols(),
                lattice_vectors=frame.get_cell().array * ureg.angstrom,
                periodic=frame.pbc,
            )
        )
        run.calculation.append(calc)
        run.system.append(system)

    archive.run.append(run)

    archive.workflow2 = EquationOfState()
    run_normalize(archive)

    return archive


def test_eos_workflow():
    archive = parse_trajectory("tests/data/ase/Cu.traj")

    eos_fit = archive.workflow2.results.eos_fit
    assert len(eos_fit) == 5
    assert eos_fit[0].fitted_energies[1].to("eV").magnitude == approx(-0.00636507)
    assert eos_fit[1].function_name == "pourier_tarantola"
    assert eos_fit[2].equilibrium_volume.to("angstrom**3").magnitude == approx(
        11.565388081047471
    )
    assert eos_fit[3].equilibrium_energy.to("eV").magnitude == approx(
        -0.007035923370513912
    )
    assert eos_fit[4].rms_error == approx(1.408202378222592e-07)


class TestSimulationWorkflow:
    """
    Tests for the base simulation workflow class.
    """

    n_calc = 10

    @pytest.fixture(autouse=True)
    def serial_simulation(self) -> EntryArchive:
        """
        Simulation with calculations done in serial.
        """
        archive = EntryArchive()
        archive.metadata = EntryMetadata(entry_type="Workflow")
        archive.workflow2 = SimulationWorkflow(
            method=SimulationWorkflowMethod(), results=SimulationWorkflowResults()
        )

        archive.run.append(
            Run(
                calculation=[
                    Calculation(time_physical=t, time_calculation=1)
                    for t in range(1, self.n_calc + 1)
                ]
            )
        )
        return archive

    def test_tasks_serial(self, serial_simulation):
        """
        Test tasks creation of purely serial calculation.
        """
        workflow = serial_simulation.workflow2
        workflow.normalize(serial_simulation, LOGGER)
        assert len(workflow.tasks) == self.n_calc

        assert workflow.inputs[0].section == workflow.tasks[0].inputs[0].section
        for n, task in enumerate(workflow.tasks[:-1]):
            assert task.name == f"Step {n + 1}"
            assert len(task.inputs) == 1
            assert len(task.outputs) == 1
            assert task.outputs[0].section == workflow.tasks[n + 1].inputs[0].section
        assert workflow.outputs[0].section == workflow.tasks[-1].outputs[0].section

    def test_tasks_defined(self, serial_simulation):
        """
        Test tasks creation skipped if tasks are predefined
        """
        workflow = SimulationWorkflow(tasks=[Task(name="1")])
        serial_simulation.workflow2 = workflow
        workflow.normalize(serial_simulation, LOGGER)

        assert len(serial_simulation.workflow2.tasks) == 1
        assert serial_simulation.workflow2.tasks[0].name == "1"

    def test_tasks_no_time(self, serial_simulation):
        """
        Test tasks creation skipped if at least one calculation has no time info.
        """

        for key in ["time_physical", "time_calculation"]:
            calc = serial_simulation.run[0].calculation[
                random.randint(0, self.n_calc - 1)
            ]
            calc.m_set(calc.m_get_quantity_definition(key), None)
            serial_simulation.workflow2.normalize(serial_simulation, LOGGER)
            assert not serial_simulation.workflow2.tasks

    @pytest.mark.parametrize(
        "calculation_indices",
        [
            # parallel (0 to 3), 4, 5, parallel (6 to 9)
            [[0, 1, 2, 3], [4], [5], [6, 7, 8, 9]],
            # 0, parallel (1 to 2), 4, 5, 6, 7, 8, 8
            [[0], [1, 2], [3], [4], [5], [6], [7], [8], [9]],
            # parallel (0 to 8), 9
            [[0, 1, 2, 3, 4, 5, 6, 7, 8], [9]],
        ],
    )
    def test_task_not_serial(self, serial_simulation, calculation_indices):
        """
        Test creation for mixed serial and parallel tasks.
        """

        def _create_times(indices, start_time=0):
            times = []
            for n in indices:
                if not isinstance(n, int):
                    times.extend(
                        sorted(
                            _create_times(n, start_time=times[-1][1] if times else 0),
                            key=lambda x: x[1],
                        )
                    )
                else:
                    calc_time = random.random()
                    dt = random.random() * 0.1  # small perturbation
                    times.append([n, calc_time + start_time + dt, calc_time])
            return times

        for n, time_physical, time_calculation in _create_times(calculation_indices):
            serial_simulation.run[-1].calculation[n].time_physical = time_physical
            serial_simulation.run[-1].calculation[n].time_calculation = time_calculation

        workflow = serial_simulation.workflow2
        workflow.normalize(serial_simulation, LOGGER)
        assert len(workflow.tasks) == 10

        # workflow inputs as inputs to first parallel tasks
        for n in calculation_indices[0]:
            assert workflow.tasks[n].name == "Step 1"
            assert workflow.tasks[n].inputs[0].section == workflow.inputs[0].section

        # outputs of previous tasks are inputs of succeeding tasks in series
        for i in range(1, len(calculation_indices)):
            for n1 in calculation_indices[i]:
                assert workflow.tasks[n1].name == f"Step {i + 1}"
                inputs = [input.section for input in workflow.tasks[n1].inputs]
                for n0 in calculation_indices[i - 1]:
                    assert workflow.tasks[n0].outputs[-1].section in inputs

        # last parallel tasks oututs as workflow outputs
        for n in calculation_indices[-1]:
            assert workflow.tasks[n].outputs[0].section in [
                output.section for output in workflow.outputs
            ]


class TestChemicalReactionWorkflow:
    """
    Contains tests for the matinfo defintion and normalization of the chemical reaction
    workflow.
    """

    @pytest.fixture(autouse=True, scope="class")
    def dft_archives(self):
        """
        Parse all relevant dft calculations.
        """
        test_dir = "tests/data/chemical_reaction"

        archives = {}
        for root, _, names in os.walk(test_dir):
            for filename in names:
                if filename not in ["vasprun.xml", "OUTCAR"]:
                    continue
                archives[os.path.basename(root)] = run_parsing(
                    VASPParser, os.path.join(root, filename)
                )
        return archives

    @pytest.fixture(autouse=True)
    def segregation_workflow_archive(self, dft_archives):
        """
        Constructs a chemical reaction workflow archive describing the segregation of H
        from RhCu_CH4 into RhCu_CH3 and RhCu_H through a transition state RhCu_CH3_H.
        """
        formula_type = [
            ["RhCu_CH4", "reactant"],
            ["RhCu", "reactant"],
            ["RhCu_CH3_H", "transition state"],
            ["RhCu_CH3", "product"],
            ["RhCu_xHfcc", "product"],
        ]
        workflow = ChemicalReaction()
        for formula, type in formula_type:
            archive = dft_archives[formula]
            workflow.inputs.append(
                Link(name=f"{formula} {type}", section=archive.run[0].calculation[-1])
            )
            # add also slab to transition state to preserve mass balance
            if formula == "RhCu":
                workflow.inputs.append(
                    Link(
                        name=f"transition state {formula}",
                        section=archive.run[0].calculation[-1],
                    )
                )

        return EntryArchive(
            metadata=EntryMetadata(entry_type="Workflow"), workflow2=workflow
        )

    @pytest.fixture(autouse=True)
    def adsorption_workflow_archive(self, dft_archives):
        """
        Constructs a chemical reaction workflow archive describing the adsorption of N
        in PdAg.
        """

        formula_type = [
            ["N", "reactant"],
            ["PdAg", "reactant"],
            ["NPdAg", "product"],
        ]
        workflow = ChemicalReaction()
        for formula, type in formula_type:
            archive = dft_archives[formula]
            workflow.inputs.append(
                Link(name=f"{formula} {type}", section=archive.run[0].calculation[-1])
            )

        return EntryArchive(
            metadata=EntryMetadata(entry_type="Workflow"), workflow2=workflow
        )

    @pytest.mark.parametrize(
        "workflow_archive, reaction_energy, activation_energy",
        [
            pytest.param(
                "segregation_workflow_archive",
                4.41467915e-20,
                1.02994872e-19,
                id="segregation",
            ),
            pytest.param(
                "adsorption_workflow_archive", -3.04029682e-19, None, id="adsorption"
            ),
        ],
    )
    def test_reaction_energy(
        self, request, workflow_archive, reaction_energy, activation_energy
    ):
        """
        Test the calculation of reaction and activation energy.
        """
        workflow_archive = request.getfixturevalue(workflow_archive)
        workflow = workflow_archive.workflow2
        workflow.normalize(workflow_archive, LOGGER)

        assert np.isclose(
            workflow.results.reaction_energy.magnitude,
            reaction_energy,
            atol=0,
            rtol=1e6,
        )
        if activation_energy:
            assert np.isclose(
                workflow.results.activation_energy.magnitude,
                activation_energy,
                atol=0,
                rtol=1e6,
            )
        assert len(workflow.tasks) == 1

    def test_system_checks(self, segregation_workflow_archive):
        """
        Test the checks for the consistency of the system from reactants
        """
        workflow = segregation_workflow_archive.workflow2
        # change the system size in an input to make them inconsistent
        lattice = np.array(workflow.inputs[-1].section.system_ref.atoms.lattice_vectors)
        workflow.inputs[-1].section.system_ref.atoms.lattice_vectors = np.ones((3, 3))
        workflow.normalize(segregation_workflow_archive, LOGGER)
        assert workflow.results.reaction_energy is None

        # change the chemical composition in an input to make them inconsistent
        workflow.inputs[-1].section.system_ref.atoms.lattice_vectoprs = lattice
        workflow.inputs[0].section.system_ref.atoms.labels = ["C"]
        workflow.normalize(segregation_workflow_archive, LOGGER)
        assert workflow.results.reaction_energy is None
