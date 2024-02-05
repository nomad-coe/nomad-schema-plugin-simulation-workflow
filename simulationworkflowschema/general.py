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
from typing import List
import numpy as np

from nomad.datamodel.data import ArchiveSection
from nomad.metainfo import SubSection, Section, Quantity, Reference
from nomad.datamodel.metainfo.common import FastAccess
from nomad.datamodel.metainfo.workflow import Workflow, Link, Task
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
)


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


INPUT_SYSTEM_NAME = "Input system"
OUTPUT_SYSTEM_NAME = "Output system"
INPUT_METHOD_NAME = "Input method"
INPUT_CALCULATION_NAME = "Input calculation"
OUTPUT_CALCULATION_NAME = "Output calculation"
WORKFLOW_METHOD_NAME = "Workflow method"
WORKFLOW_RESULTS_NAME = "Workflow results"


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
        shape=["n_calculations"],
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

        self._calculations: List[Calculation] = []
        self._systems: List[System] = []
        self._methods: List[Method] = []
        try:
            self._calculations = archive.run[-1].calculation
            self._systems = archive.run[-1].system
            self._methods = archive.run[-1].method
        except Exception:
            logger.warning("System, method and calculation required for normalization.")
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
                task.name = f"Step {current_step}"
                tasks.append(task)
                current_time = max(current_time, calc.time_physical)
                # add input if first calc in tasks
                add_inputs = add_inputs or n == 0
                # add output if last calc in tasks
                add_outputs = add_outputs or n == len(self._calculations) - 1
            for task in tasks:
                # add workflow inputs to first parallel tasks
                if task.name == "Step 1" and add_inputs:
                    task.inputs.extend(
                        [input for input in self.inputs if input not in task.inputs]
                    )
                # add outputs of last parallel tasks to workflow outputs
                if task.name == f"Step {current_step}" and add_outputs:
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
                    Task(name=f"Calculation {n}", inputs=inputs, outputs=outputs)
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
                    Task(name=f"Step {n}", inputs=inputs, outputs=outputs)
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
            name = f"{name}_electronic" if name in ["dos", "band_structure"] else name
            try:
                calc_section = getattr(task.outputs[-1].section, name)
                if calc_section:
                    output_section.m_set(section, calc_section)
            except Exception:
                continue

    def get_gw_workflow_results(
        self, initial_task: TaskReference, final_task: TaskReference
    ) -> None:
        """
        Gets the GW workflow results section by resolving the DFTOutputs from the initial
        task and the GWOutputs from the final task.
        """
        dft_outputs = ElectronicStructureOutputs()
        self._resolve_outputs_section(dft_outputs, initial_task)
        gw_outputs = ElectronicStructureOutputs()
        self._resolve_outputs_section(gw_outputs, final_task)
        self.results.dft_outputs = dft_outputs
        self.results.gw_outputs = gw_outputs

    def get_dmft_workflow_results(
        self, initial_task: TaskReference, final_task: TaskReference
    ) -> None:
        """
        Gets the DMFT workflow results section by resolving the TBOutputs from the initial
        task and the DMFTOutputs from the final task.
        """
        tb_outputs = ElectronicStructureOutputs()
        self._resolve_outputs_section(tb_outputs, initial_task)
        dmft_outputs = ElectronicStructureOutputs()
        self._resolve_outputs_section(dmft_outputs, final_task)
        self.results.tb_outputs = tb_outputs
        self.results.dmft_outputs = dmft_outputs

    def get_maxent_workflow_results(
        self, initial_task: TaskReference, final_task: TaskReference
    ) -> None:
        """
        Gets the MaxEnt workflow results section by resolving the DMFTOutputs from the
        initial task and the MaxEntOutputs from the final task.
        """
        dmft_outputs = ElectronicStructureOutputs()
        self._resolve_outputs_section(dmft_outputs, initial_task)
        maxent_outputs = ElectronicStructureOutputs()
        self._resolve_outputs_section(maxent_outputs, final_task)
        self.results.dmft_outputs = dmft_outputs
        self.results.maxent_outputs = maxent_outputs

    def get_tb_workflow_results(
        self, initial_task: TaskReference, final_task: TaskReference
    ) -> None:
        """
        Gets the TB workflow results section by resolving the FirstPrinciplesOutputs from the
        initial task and the TBOutputs from the final task.
        """
        first_principles_outputs = ElectronicStructureOutputs()
        self._resolve_outputs_section(first_principles_outputs, initial_task)
        tb_outputs = ElectronicStructureOutputs()
        self._resolve_outputs_section(tb_outputs, final_task)
        self.results.first_principles_outputs = first_principles_outputs
        self.results.tb_outputs = tb_outputs

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if len(self.tasks) != 2:
            logger.error("Expected two tasks.")
            return

        initial_task = self.tasks[0]
        final_task = self.tasks[1]

        workflow_name = self.m_def.name.replace("Plus", "+")
        self.name = workflow_name
        if workflow_name == "DFT+GW":
            self.get_gw_workflow_results(initial_task, final_task)
        elif workflow_name == "DFT+TB+DMFT":
            self.get_dmft_workflow_results(initial_task, final_task)
        elif workflow_name == "DMFT+MaxEnt":
            self.get_maxent_workflow_results(initial_task, final_task)
        elif workflow_name == "FirstPrinciples+TB":
            self.get_tb_workflow_results(initial_task, final_task)


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
        shape=["*"],
        description="""
        Reference to the band gap section.
        """,
    )

    dos = Quantity(
        type=Reference(Dos),
        shape=["*"],
        description="""
        Reference to the density of states section.
        """,
    )

    band_structure = Quantity(
        type=Reference(BandStructure),
        shape=["*"],
        description="""
        Reference to the band structure section.
        """,
    )

    greens_functions = Quantity(
        type=Reference(GreensFunctions),
        shape=["*"],
        description="""
        Ref to the Green functions section.
        """,
    )
