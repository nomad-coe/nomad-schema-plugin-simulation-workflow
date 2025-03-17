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

from nomad.datamodel.data import ArchiveSection
from nomad.metainfo import SubSection, Section, Quantity, Reference, Package
from nomad.datamodel.metainfo.common import FastAccess
from nomad.datamodel.metainfo.workflow import Workflow, Link, Task, TaskReference
from runschema.method import (
    Method,
    XCFunctional,
    BasisSetContainer,
)
from runschema.system import System
from runschema.calculation import (
    Calculation,
    BandGap,
    Dos,
    BandStructure,
    GreensFunctions,
    MagneticShielding,
    ElectricFieldGradient,
    SpinSpinCoupling,
    MagneticSusceptibility,
)


m_package = Package()


def resolve_difference(values):
    delta_values = None

    for n in range(-1, -len(values), -1):
        a = values[n]
        b = values[n - 1]
        if a is None or b is None:
            continue
        delta_values = abs(a - b)
        if delta_values != 0.0:
            break

    return delta_values


INPUT_SYSTEM_NAME = 'Input system'
OUTPUT_SYSTEM_NAME = 'Output system'
INPUT_METHOD_NAME = 'Input method'
INPUT_CALCULATION_NAME = 'Input calculation'
OUTPUT_CALCULATION_NAME = 'Output calculation'
WORKFLOW_METHOD_NAME = 'Workflow method'
WORKFLOW_RESULTS_NAME = 'Workflow results'


class SimulationWorkflowMethod(ArchiveSection):
    m_def = Section(validate=False)

    def normalize(self, archive, logger):
        pass


class SimulationWorkflowResults(ArchiveSection):
    m_def = Section(validate=False)

    calculation_result_ref = Quantity(
        type=Reference(Calculation.m_def),
        shape=[],
        description="""
        Reference to calculation result. In the case of serial workflows, this corresponds
        to the final step in the simulation. For the parallel case, it refers to the reference calculation.
        """,
        categories=[FastAccess],
    )

    n_calculations = Quantity(
        type=int,
        shape=[],
        description="""
        Number of calculations in workflow.
        """,
    )

    calculations_ref = Quantity(
        type=Reference(Calculation.m_def),
        shape=['n_calculations'],
        description="""
        List of references to each calculation section in the simulation.
        """,
    )

    def normalize(self, archive, logger):
        calculations = []
        try:
            calculations = archive.run[0].calculation
        except Exception:
            return

        if not calculations:
            return

        if not self.calculation_result_ref:
            self.calculation_result_ref = calculations[-1]

        if not self.calculations_ref:
            self.calculations_ref = calculations


class SimulationWorkflow(Workflow):
    method = SubSection(sub_section=SimulationWorkflowMethod)

    results = SubSection(sub_section=SimulationWorkflowResults, categories=[FastAccess])

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        self._calculations: list[ArchiveSection] = []  # noqa
        self._systems: list[ArchiveSection] = []  # noqa
        self._methods: list[ArchiveSection] = []  # noqa
        try:
            self._calculations = archive.run[-1].calculation
            self._systems = archive.run[-1].system
            self._methods = archive.run[-1].method
        except Exception:
            logger.warning('System, method and calculation required for normalization.')
            pass

        if not self._calculations:
            return

        if not self.inputs:
            if self._systems:
                self.m_add_sub_section(
                    Workflow.inputs,
                    Link(name=INPUT_SYSTEM_NAME, section=self._systems[0]),
                )

            if self.method is not None:
                self.m_add_sub_section(
                    Workflow.inputs,
                    Link(name=WORKFLOW_METHOD_NAME, section=self.method),
                )

        for link in self.inputs:
            if isinstance(link.section, System):
                self.input_structure = link.section
                break

        if not self.outputs:
            if self._calculations:
                self.m_add_sub_section(
                    Workflow.outputs,
                    Link(name=OUTPUT_CALCULATION_NAME, section=self._calculations[-1]),
                )

            if self.results is not None:
                self.m_add_sub_section(
                    Workflow.outputs,
                    Link(name=WORKFLOW_RESULTS_NAME, section=self.results),
                )

        if not self.tasks:
            current_step = 1
            current_time = None
            inputs = []
            add_inputs, add_outputs = False, False
            tasks = []
            for n, calc in enumerate(self._calculations):
                if calc.time_physical is None or calc.time_calculation is None:
                    # all calculation sections should have time info inorder to create workflow tasks
                    tasks = []
                    break

                task = Task(outputs=[Link(name=OUTPUT_CALCULATION_NAME, section=calc)])
                # successive tasks in serial if overlap in time duration exists
                if current_time is None:
                    current_time = calc.time_physical
                start_time = calc.time_physical - calc.time_calculation
                if start_time and (
                    start_time > current_time or np.isclose(start_time, current_time)
                ):
                    task.inputs = inputs
                    inputs = []
                    current_step += 1
                else:
                    if tasks:
                        for input in tasks[-1].inputs:
                            if input.name == INPUT_METHOD_NAME:
                                continue
                            task.m_add_sub_section(Task.inputs, input)
                if calc.method_ref:
                    task.m_add_sub_section(
                        Task.inputs,
                        Link(name=INPUT_METHOD_NAME, section=calc.method_ref),
                    )
                if calc.system_ref:
                    inputs.append(Link(name=INPUT_SYSTEM_NAME, section=calc.system_ref))
                    task.m_add_sub_section(
                        Task.outputs,
                        Link(name=OUTPUT_SYSTEM_NAME, section=calc.system_ref),
                    )
                else:
                    inputs.append(Link(name=INPUT_CALCULATION_NAME, section=calc))
                task.name = f'Step {current_step}'
                tasks.append(task)
                current_time = max(current_time, calc.time_physical)
                # add input if first calc in tasks
                add_inputs = add_inputs or n == 0
                # add output if last calc in tasks
                add_outputs = add_outputs or n == len(self._calculations) - 1
            for task in tasks:
                # add workflow inputs to first parallel tasks
                if task.name == 'Step 1' and add_inputs:
                    task.inputs.extend(
                        [input for input in self.inputs if input not in task.inputs]
                    )
                # add outputs of last parallel tasks to workflow outputs
                if task.name == f'Step {current_step}' and add_outputs:
                    self.outputs.extend(
                        [
                            output
                            for output in task.outputs
                            if output not in self.outputs
                        ]
                    )

            self.tasks = tasks


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


class SerialSimulation(SimulationWorkflow):
    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if not self.tasks:
            previous_structure = None
            for n, calculation in enumerate(self._calculations):
                inputs, outputs = (
                    [],
                    [Link(name=OUTPUT_CALCULATION_NAME, section=calculation)],
                )
                if calculation.system_ref:
                    input_structure = (
                        self.input_structure
                        if n == 0
                        else self._calculations[n - 1].system_ref
                    )
                    if not input_structure:
                        input_structure = previous_structure
                    if input_structure:
                        inputs.append(
                            Link(name=INPUT_SYSTEM_NAME, section=input_structure)
                        )
                    previous_structure = calculation.system_ref
                    outputs.append(
                        Link(name=OUTPUT_SYSTEM_NAME, section=calculation.system_ref)
                    )
                elif len(self._calculations) == len(self._systems):
                    inputs.append(
                        Link(
                            name=INPUT_SYSTEM_NAME,
                            section=self.input_structure
                            if n == 0
                            else self._systems[n - 1],
                        )
                    )
                    outputs.append(
                        Link(name=OUTPUT_SYSTEM_NAME, section=self._systems[n])
                    )
                else:
                    continue
                if calculation.method_ref:
                    inputs.append(
                        Link(name=INPUT_METHOD_NAME, section=calculation.method_ref)
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
                    Task(name=f'Step {n}', inputs=inputs, outputs=outputs)
                )


class BeyondDFT(SerialSimulation):
    """
    Base class used to normalize standard workflows beyond DFT containing two specific
    SinglePoint tasks (GWWorkflow = DFT + GW, DMFTWorkflow = DFT + DMFT,
    MaxEntWorkflow = DMFT + MaxEnt, and so on) and store the outputs in the self.results
    section.
    """

    def _resolve_outputs_section(self, output_section, task: TaskReference) -> None:
        """
        Resolves the output_section of a task and stores the results in the output_section.

        Args:
            task (TaskReference): The task from which the outputs are got.
        """
        for name, section in output_section.m_def.all_quantities.items():
            name = f'{name}_electronic' if name in ['dos', 'band_structure'] else name
            try:
                calc_section = getattr(task.outputs[-1].section, name)
                if calc_section:
                    output_section.m_set(section, calc_section)
            except Exception:
                continue

    def get_electronic_structure_workflow_results(self, task_map: dict) -> None:
        """
        Gets the standard electronic structure workflow results section by resolving the
        outputs specified in the `task_map`.

        Args:
            task_map (dict): The dictionary used to resolve the outputs sections.
        """
        for method, task in task_map.items():
            outputs = ElectronicStructureOutputs()
            self._resolve_outputs_section(outputs, task)
            setattr(self.results, f'{method}_outputs', outputs)

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if len(self.tasks) != 2:
            logger.error('Expected two tasks.')
            return

        # We extract the workflow name from the tasks names
        self.name = '+'.join([task.name for task in self.tasks if task.name])
        task_map = {
            task.name.lower(): self.tasks[n] for n, task in enumerate(self.tasks)
        }
        # Resolve workflow2.results for each standard BeyondDFT workflow
        if self.name == 'DFT+GW':
            self.get_electronic_structure_workflow_results(task_map)
        elif self.name == 'TB+DMFT':  # TODO extend for DFT tasks
            self.get_electronic_structure_workflow_results(task_map)
        elif self.name == 'DMFT+MaxEnt':
            self.get_electronic_structure_workflow_results(task_map)
        elif self.name == 'FirstPrinciples+TB':
            task_map['first_principles'] = task_map.pop('firstprinciples')
            self.get_electronic_structure_workflow_results(task_map)


class DFTMethod(SimulationWorkflowMethod):
    """
    Base class defining the DFT input methodologies: starting XC functional and electrons
    representation (basis set).
    """

    starting_point = Quantity(
        type=Reference(XCFunctional),
        description="""
        Reference to the starting point (XC functional or HF) used.
        """,
    )

    electrons_representation = Quantity(
        type=Reference(BasisSetContainer),
        description="""
        Reference to the basis set used.
        """,
    )


class ElectronicStructureOutputs(SimulationWorkflowResults):
    """
    Base class defining the typical output properties of any electronic structure
    SinglePoint calculation: DFT, TB, DMFT, GW, MaxEnt, XS.
    """

    band_gap = Quantity(
        type=Reference(BandGap),
        shape=['*'],
        description="""
        Reference to the band gap section.
        """,
    )

    dos = Quantity(
        type=Reference(Dos),
        shape=['*'],
        description="""
        Reference to the density of states section.
        """,
    )

    band_structure = Quantity(
        type=Reference(BandStructure),
        shape=['*'],
        description="""
        Reference to the band structure section.
        """,
    )

    greens_functions = Quantity(
        type=Reference(GreensFunctions),
        shape=['*'],
        description="""
        Ref to the Green functions section.
        """,
    )


class MagneticOutputs(SimulationWorkflowResults):
    """
    Base class defining the typical output properties of magnetic SinglePoint calculations.
    """

    magnetic_shielding = Quantity(
        type=Reference(MagneticShielding),
        shape=['*'],
        description="""
        Reference to the magnetic shielding tensors.
        """,
    )

    electric_field_gradient = Quantity(
        type=Reference(ElectricFieldGradient),
        shape=['*'],
        description="""
        Reference to the electric field gradient tensors.
        """,
    )

    spin_spin_coupling = Quantity(
        type=Reference(SpinSpinCoupling),
        shape=['*'],
        description="""
        Reference to the spin-spin coupling tensors.
        """,
    )

    magnetic_susceptibility_nmr = Quantity(
        type=Reference(MagneticSusceptibility),
        shape=['*'],
        description="""
        Reference to the magnetic susceptibility tensors.
        """,
    )


m_package.__init_metainfo__()
