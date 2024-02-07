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
from nomad.metainfo import SubSection, Quantity, Reference
from runschema.method import TB as TBMethodology, Method
from .general import (
    SimulationWorkflowResults,
    ElectronicStructureOutputs,
    SimulationWorkflowMethod,
    SerialSimulation,
)


class FirstPrinciplesPlusTBResults(SimulationWorkflowResults):
    """
    Groups first principles and TB outputs: band gaps, DOS, band structures. The
    ResultsNormalizer takes care of adding a label 'FirstPrinciples' or 'TB' in the method
    `get_tb_workflow_properties`.
    """

    first_principles_outputs = SubSection(
        sub_section=ElectronicStructureOutputs.m_def, repeats=False
    )

    tb_outputs = SubSection(sub_section=ElectronicStructureOutputs.m_def, repeats=False)


class FirstPrinciplesPlusTBMethod(SimulationWorkflowMethod):
    """
    Specifies both the first principles and the TB input methodologies.
    """

    # TODO refine this referencing
    first_principles_method_ref = Quantity(
        type=Reference(Method),
        description="""
        First principles methodology reference.
        """,
    )

    tb_method_ref = Quantity(
        type=Reference(TBMethodology),
        description="""
        TB methodology reference.
        """,
    )


class FirstPrinciplesPlusTB(SerialSimulation):
    """
    The TB (tight-binding) workflow is generated in an extra EntryArchive IF both
    the first principles SinglePoint and the TB SinglePoint EntryArchives are present in the upload.
    """

    method = SubSection(sub_section=FirstPrinciplesPlusTBMethod)

    results = SubSection(sub_section=FirstPrinciplesPlusTBResults)

    def normalize(self, archive, logger):
        if not self.results:  # creates Results section if not present
            self.results = FirstPrinciplesPlusTBResults()

        super().normalize(archive, logger)
