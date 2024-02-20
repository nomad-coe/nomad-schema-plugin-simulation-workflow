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

from nomad.metainfo import SubSection, Quantity, Reference
from nomad.datamodel.metainfo.workflow import Link, Task
from runschema.calculation import (
    Dos,
    BandStructure,
    BandEnergies,
    Density,
    Potential,
    Spectra,
)
from .general import (
    SimulationWorkflowResults,
    SimulationWorkflowMethod,
    SimulationWorkflow,
    INPUT_SYSTEM_NAME,
    INPUT_METHOD_NAME,
    OUTPUT_CALCULATION_NAME,
    WORKFLOW_METHOD_NAME,
    WORKFLOW_RESULTS_NAME,
    resolve_difference,
)


class SinglePointResults(SimulationWorkflowResults):
    n_scf_steps = Quantity(
        type=int,
        shape=[],
        description="""
        Number of self-consistent steps in the calculation.
        """,
    )

    final_scf_energy_difference = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        The difference in the energy between the last two scf steps.
        """,
    )

    is_converged = Quantity(
        type=bool,
        shape=[],
        description="""
        Indicates if the convergence criteria were fullfilled.
        """,
    )

    n_data = Quantity(
        type=np.int32,
        shape=[],
        description="""
        """,
    )

    dos = Quantity(
        type=Reference(Dos),
        shape=['n_data'],
        description="""
        Reference to the electronic density of states data.
        """,
    )

    band_structure = Quantity(
        type=Reference(BandStructure),
        shape=['n_data'],
        description="""
        Reference to the electronic band structure data.
        """,
    )

    eigenvalues = Quantity(
        type=Reference(BandEnergies),
        shape=['n_data'],
        description="""
        Reference to the eigenvalues.
        """,
    )

    potential = Quantity(
        type=Reference(Potential),
        shape=['n_data'],
        description="""
        Reference to the potential data.
        """,
    )

    density_charge = Quantity(
        type=Reference(Density),
        shape=['n_data'],
        description="""
        Reference to the charge density data.
        """,
    )

    spectra = Quantity(
        type=Reference(Spectra),
        shape=['n_data'],
        description="""
        Reference to the spectral data.
        """,
    )


class SinglePointMethod(SimulationWorkflowMethod):
    method = Quantity(
        type=str,
        shape=[],
        description="""
        Calculation method used.
        """,
    )


class SinglePoint(SimulationWorkflow):
    method = SubSection(sub_section=SinglePointMethod)

    results = SubSection(sub_section=SinglePointResults)

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if not self.tasks:
            task = Task()
            if self._systems:
                task.m_add_sub_section(
                    Task.inputs, Link(name=INPUT_SYSTEM_NAME, section=self._systems[0])
                )
            if self._methods:
                task.m_add_sub_section(
                    Task.inputs, Link(name=INPUT_METHOD_NAME, section=self._methods[0])
                )
            if self._calculations:
                task.m_add_sub_section(
                    Task.inputs,
                    Link(name=OUTPUT_CALCULATION_NAME, section=self._calculations[0]),
                )

            self.tasks = [task]

        if not self.method:
            self.method = SinglePointMethod()

        if not self.inputs:
            self.m_add_sub_section(
                SimulationWorkflow.inputs,
                Link(name=WORKFLOW_METHOD_NAME, section=self.method),
            )

        if not self.results:
            self.results = SinglePointResults()

        if not self.outputs:
            self.m_add_sub_section(
                SimulationWorkflow.outputs,
                Link(name=WORKFLOW_RESULTS_NAME, section=self.results),
            )

        if not self.method.method:
            try:
                # TODO keep extending for other SinglePoint
                for method_name in ['dft', 'gw', 'bse', 'dmft']:
                    if self._methods[-1].m_xpath(method_name):
                        self.method.method = method_name.upper()
                        break
            except Exception:
                pass

        if not self._calculations:
            return

        last_calc = self._calculations[-1]
        if not self.results.n_scf_steps:
            self.results.n_scf_steps = len(last_calc.scf_iteration)

        energies = [
            scf.energy.total.value
            for scf in last_calc.scf_iteration
            if scf.energy is not None and scf.energy.total is not None
        ]
        delta_energy = resolve_difference(energies)
        if not self.results.final_scf_energy_difference and delta_energy is not None:
            self.results.final_scf_energy_difference = delta_energy

        if not self.results.is_converged and delta_energy is not None:
            try:
                threshold = self._methods[-1].scf.threshold_energy_change
                self.results.is_converged = bool(delta_energy <= threshold)
            except Exception:
                pass

        if not self.results.dos and last_calc.dos_electronic:
            self.results.dos = last_calc.dos_electronic

        if not self.results.band_structure and last_calc.band_structure_electronic:
            self.results.band_structure = last_calc.band_structure_electronic

        if not self.results.eigenvalues and last_calc.eigenvalues:
            self.results.eigenvalues = last_calc.eigenvalues

        if not self.results.density_charge and last_calc.density_charge:
            self.results.density_charge = last_calc.density_charge

        if not self.results.potential and last_calc.potential:
            self.results.potential = last_calc.potential

        if not self.results.spectra and last_calc.spectra:
            self.results.spectra = last_calc.spectra

        SimulationWorkflowResults.normalize(self.results, archive, logger)


class ParallelSimulation(SimulationWorkflow):
    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if not self.tasks:
            for n, calculation in enumerate(self._calculations):
                inputs, outputs = (
                    [],
                    [Link(name=OUTPUT_CALCULATION_NAME, section=calculation)],
                )
                if self._calculations[n].system_ref:
                    inputs.append(
                        Link(
                            name=INPUT_SYSTEM_NAME,
                            section=self._calculations[n].system_ref,
                        )
                    )
                elif len(self._calculations) == len(self._systems):
                    inputs.append(
                        Link(name=INPUT_SYSTEM_NAME, section=self._systems[n])
                    )
                else:
                    continue
                if self._calculations[n].method_ref:
                    inputs.append(
                        Link(
                            name=INPUT_METHOD_NAME,
                            section=self._calculations[n].method_ref,
                        )
                    )
                elif len(self._calculations) == len(self._methods):
                    inputs.append(
                        Link(name=INPUT_METHOD_NAME, section=self._methods[n])
                    )
                elif len(self._methods) == 1:
                    inputs.append(
                        Link(name=INPUT_METHOD_NAME, section=self._methods[0])
                    )
                self.tasks.append(
                    Task(name=f'Calculation {n}', inputs=inputs, outputs=outputs)
                )
