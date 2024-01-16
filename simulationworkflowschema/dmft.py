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
from runschema.method import (
    TB as TBMethodology,
    DMFT as DMFTMethodology,
)
from .general import (
    SimulationWorkflowResults,
    ElectronicStructureOutputs,
    DFTMethod,
    BeyondDFT,
)


class DFTPlusTBPlusDMFTResults(SimulationWorkflowResults):
    """
    Groups DFT, TB and DMFT outputs: band gaps (all), DOS (DFT, TB), band
    structures (DFT, TB), Greens functions (DMFT). The ResultsNormalizer takes care
    of adding a label 'DFT', 'PROJECTION, or 'DMFT' in the method `get_dmft_workflow_properties`.
    """

    dft_outputs = SubSection(
        sub_section=ElectronicStructureOutputs.m_def, repeats=False
    )

    tb_outputs = SubSection(sub_section=ElectronicStructureOutputs.m_def, repeats=False)

    dmft_outputs = SubSection(
        sub_section=ElectronicStructureOutputs.m_def, repeats=False
    )


class DFTPlusTBPlusDMFTMethod(DFTMethod):
    """
    Specifies all DFT, TB and DMFT input methodologies: starting XC functional, electrons
    representation (basis set), TB method reference, DMFT method reference.
    """

    tb_method_ref = Quantity(
        type=Reference(TBMethodology),
        description="""
        TB methodology reference.
        """,
    )

    dmft_method_ref = Quantity(
        type=Reference(DMFTMethodology),
        description="""
        DMFT methodology reference.
        """,
    )


class DFTPlusTBPlusDMFT(BeyondDFT):  # TODO implement connection with DFT task
    """
    The DMFT workflow is generated in an extra EntryArchive IF both the TB SinglePoint
    and the DMFT SinglePoint EntryArchives are present in the upload.
    """

    method = SubSection(sub_section=DFTPlusTBPlusDMFTMethod)

    results = SubSection(sub_section=DFTPlusTBPlusDMFTResults)

    def normalize(self, archive, logger):
        if not self.results:  # creates Results section if not present
            self.results = DFTPlusTBPlusDMFTResults()

        super().normalize(archive, logger)
