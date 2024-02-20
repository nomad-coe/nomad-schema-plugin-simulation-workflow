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
from ase.eos import EquationOfState as aseEOS

from nomad.atomutils import get_volume
from nomad.datamodel.data import ArchiveSection
from nomad.units import ureg
from nomad.metainfo import SubSection, Section, Quantity
from nomad.datamodel.metainfo.workflow import Link
from .general import (
    SimulationWorkflowMethod,
    SimulationWorkflowResults,
    ParallelSimulation,
    WORKFLOW_METHOD_NAME,
    WORKFLOW_RESULTS_NAME,
)


class EquationOfStateMethod(SimulationWorkflowMethod):
    energy_calculator = Quantity(
        type=str,
        shape=[],
        description="""
        Name of program used to calculate energy.
        """,
    )


class EOSFit(ArchiveSection):
    """
    Section containing results of an equation of state fit.
    """

    m_def = Section(validate=False)

    function_name = Quantity(
        type=str,
        shape=[],
        description="""
        Specifies the function used to perform the fitting of the volume-energy data. Value
        can be one of birch_euler, birch_lagrange, birch_murnaghan, mie_gruneisen,
        murnaghan, pack_evans_james, poirier_tarantola, tait, vinet.
        """,
    )

    fitted_energies = Quantity(
        type=np.float64,
        shape=['n_points'],
        unit='joule',
        description="""
        Array of the fitted energies corresponding to each volume.
        """,
    )

    bulk_modulus = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        Calculated value of the bulk modulus by fitting the volume-energy data.
        """,
    )

    bulk_modulus_derivative = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Calculated value of the pressure derivative of the bulk modulus.
        """,
    )

    equilibrium_volume = Quantity(
        type=np.float64,
        shape=[],
        unit='m ** 3',
        description="""
        Calculated value of the equilibrium volume by fitting the volume-energy data.
        """,
    )

    equilibrium_energy = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        Calculated value of the equilibrium energy by fitting the volume-energy data.
        """,
    )

    rms_error = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Root-mean squared value of the error in the fitting.
        """,
    )


class EquationOfStateResults(SimulationWorkflowResults):
    n_points = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of volume-energy pairs in data.
        """,
    )

    volumes = Quantity(
        type=np.float64,
        shape=['n_points'],
        unit='m ** 3',
        description="""
        Array of volumes per atom for which the energies are evaluated.
        """,
    )

    energies = Quantity(
        type=np.float64,
        shape=['n_points'],
        unit='joule',
        description="""
        Array of energies corresponding to each volume.
        """,
    )

    eos_fit = SubSection(sub_section=EOSFit.m_def, repeats=True)


class EquationOfState(ParallelSimulation):
    method = SubSection(sub_section=EquationOfStateMethod)

    results = SubSection(sub_section=EquationOfStateResults)

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if not self.method:
            self.method = EquationOfStateMethod()
            self.inputs.append(Link(name=WORKFLOW_METHOD_NAME, section=self.method))

        if not self.results:
            self.results = EquationOfStateResults()
            self.outputs.append(Link(name=WORKFLOW_RESULTS_NAME, section=self.results))

        if not self._calculations:
            return

        if self.results.energies is None:
            try:
                self.results.energies = [
                    calc.energy.total.value.magnitude for calc in self._calculations
                ]
            except Exception:
                pass

        if self.results.volumes is None:
            try:
                volumes = []
                unit = 1
                for system in self._systems:
                    if system.atoms.lattice_vectors is not None:
                        cell = system.atoms.lattice_vectors.magnitude
                        unit = system.atoms.lattice_vectors.units
                        volumes.append(get_volume(cell))
                self.results.volumes = np.array(volumes) * unit**3
            except Exception:
                pass

        if not self.results.eos_fit:
            function_name_map = {
                'birch_murnaghan': 'birchmurnaghan',
                'pourier_tarantola': 'pouriertarantola',
                'vinet': 'vinet',
                'murnaghan': 'murnaghan',
                'birch_euler': 'birch',
            }
            if self.results.volumes is not None and self.results.energies is not None:
                # convert to ase units in order for function optimization to work
                volumes = self.results.volumes.to('angstrom ** 3').magnitude
                energies = self.results.energies.to('eV').magnitude
                for function_name, ase_name in function_name_map.items():
                    try:
                        eos = aseEOS(volumes, energies, ase_name)
                        eos.fit()
                        fitted_energies = eos.func(volumes, *eos.eos_parameters)
                        rms_error = np.sqrt(np.mean((fitted_energies - energies) ** 2))
                        eos_fit = EOSFit(
                            function_name=function_name,
                            fitted_energies=fitted_energies * ureg.eV,
                            bulk_modulus=eos.B * ureg.eV / ureg.angstrom**3,
                            equilibrium_volume=eos.v0 * ureg.angstrom**3,
                            equilibrium_energy=eos.e0 * ureg.eV,
                            rms_error=rms_error,
                        )
                        self.results.eos_fit.append(eos_fit)
                    except Exception:
                        self.logger.warning('EOS fit not succesful.')
