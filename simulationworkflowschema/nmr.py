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
from .general import (
    DFTOutputs,
    NMROutputs,
    DFTMethod,
    BeyondDFT2Tasks,
)


class NMRResults(DFTOutputs, NMROutputs):
    """
    Groups DFT and NMR outputs: band gaps, DOS, band structures, and magnetic outputs.
    The ResultsNormalizer takes care of adding a label 'DFT' or 'NMR' in the method
    `get_nmr_workflow_properties`.
    """

    pass


class NMRMethod(DFTMethod):
    """
    Groups DFT and NRM input methodologies: starting XC functional, electrons
    representation (basis set).
    """

    pass


class NMR(BeyondDFT2Tasks):
    """
    The NMR workflow is generated in an extra EntryArchive IF both the DFT SinglePoint
    and the NMR SinglePoint EntryArchives are present in the upload.
    """

    method = SubSection(sub_section=NMRMethod)

    results = SubSection(sub_section=NMRResults)

    def normalize(self, archive, logger):
        if len(self.tasks) != 2:
            logger.error("Expected two tasks.")
            return

        super().normalize(archive, logger)
