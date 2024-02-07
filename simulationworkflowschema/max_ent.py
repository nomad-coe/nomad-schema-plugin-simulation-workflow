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
from runschema.method import Method, DMFT as DMFTMethodology
from .general import (
    SimulationWorkflowResults,
    ElectronicStructureOutputs,
    SimulationWorkflowMethod,
    BeyondDFT,
)


class DMFTPlusMaxEntResults(SimulationWorkflowResults):
    """
    Groups DMFT and MaxEnt outputs: greens functions (DMFT, MaxEnt), band gaps (MaxEnt),
    DOS (MaxEnt), band structures (MaxEnt). The ResultsNormalizer takes care of adding a
    label 'DMFT' or 'MaxEnt' in the method `get_maxent_workflow_properties`.
    """

    dmft_outputs = SubSection(
        sub_section=ElectronicStructureOutputs.m_def, repeats=False
    )

    maxent_outputs = SubSection(
        sub_section=ElectronicStructureOutputs.m_def, repeats=False
    )


class DMFTPlusMaxEntMethod(SimulationWorkflowMethod):
    """
    Specifies both DMFT and MaxEnt input methodologies: DMFT method references, MaxEnt method
    reference.
    """

    dmft_method_ref = Quantity(
        type=Reference(DMFTMethodology),
        description="""
        DMFT methodology reference.
        """,
    )

    # TODO define MaxEnt metainfo in Method
    maxent_method_ref = Quantity(
        type=Reference(Method),
        description="""
        MaxEnt methodology reference.
        """,
    )


class DMFTPlusMaxEnt(BeyondDFT):
    """
    The MaxEnt (Maximum Entropy) workflow is generated in an extra EntryArchive IF both
    the DMFT SinglePoint and the MaxEnt SinglePoint EntryArchives are present in the upload.
    """

    method = SubSection(sub_section=DMFTPlusMaxEntMethod)

    results = SubSection(sub_section=DMFTPlusMaxEntResults)

    def normalize(self, archive, logger):
        if not self.results:  # creates Results section if not present
            self.results = DMFTPlusMaxEntResults()

        super().normalize(archive, logger)
