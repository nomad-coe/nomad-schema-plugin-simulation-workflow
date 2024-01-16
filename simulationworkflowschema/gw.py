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
from nomad.datamodel.metainfo.simulation.method import GW as GWMethodology
from .general import (
    SimulationWorkflowResults,
    DFTOutputs,
    GWOutputs,
    DFTMethod,
    BeyondDFT,
)


class GWResults(SimulationWorkflowResults):
    """
    Groups DFT and GW outputs: band gaps, DOS, band structures. The ResultsNormalizer
    takes care of adding a label 'DFT' or 'GW' in the method `get_gw_workflow_properties`.
    """

    dft_outputs = SubSection(sub_section=DFTOutputs.m_def, repeats=False)

    gw_outputs = SubSection(sub_section=GWOutputs.m_def, repeats=False)


class GWMethod(DFTMethod):
    """
    Specifies both DFT and GW input methodologies: starting XC functional, electrons
    representation (basis set), GW method reference.
    """

    gw_method_ref = Quantity(
        type=Reference(GWMethodology),
        description="""
        Reference to the GW methodology.
        """,
    )


class GW(BeyondDFT):
    """
    The GW workflow is generated in an extra EntryArchive IF both the DFT SinglePoint
    and the GW SinglePoint EntryArchives are present in the upload.
    """

    method = SubSection(sub_section=GWMethod)

    results = SubSection(sub_section=GWResults)

    def normalize(self, archive, logger):
        if not self.results:  # creates Results section if not present
            self.results = GWResults()

        super().normalize(archive, logger)
