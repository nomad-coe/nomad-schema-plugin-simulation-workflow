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
from nomad.metainfo import SubSection
from runschema.calculation import (
    Spectra,
    ElectronicStructureProvenance,
)
from .general import (
    SimulationWorkflowResults,
    ElectronicStructureOutputs,
    SimulationWorkflowMethod,
    BeyondDFT,
)
from .photon_polarization import PhotonPolarizationResults


class XSResults(SimulationWorkflowResults):
    """
    Groups DFT, GW and PhotonPolarization outputs: band gaps (DFT, GW), DOS (DFT, GW),
    band structures (DFT, GW), spectra (PhotonPolarization). The ResultsNormalizer takes
    care of adding a label 'DFT' or 'GW' in the method `get_xs_workflow_properties`.
    """

    dft_outputs = SubSection(
        sub_section=ElectronicStructureOutputs.m_def, repeats=False
    )

    gw_outputs = SubSection(sub_section=ElectronicStructureOutputs.m_def, repeats=False)

    spectra = SubSection(sub_section=PhotonPolarizationResults, repeats=True)


class XSMethod(SimulationWorkflowMethod):
    pass


class XS(BeyondDFT):
    """
    The XS workflow is generated in an extra EntryArchive IF both the DFT SinglePoint
    and the PhotonPolarization EntryArchives are present in the upload.
    """

    # TODO extend to reference a GW SinglePoint.

    method = SubSection(sub_section=XSMethod)

    results = SubSection(sub_section=XSResults)

    def normalize(self, archive, logger):
        if len(self.tasks) < 2:
            logger.error(
                'Expected more than one task: DFT+PhotonPolarization or DFT+GW+PhotonPolarization.'
            )
            return

        dft_task = self.tasks[0]
        xs_tasks = [self.tasks[i] for i in range(1, len(self.tasks))]
        gw_task = None
        # Check if the xs_tasks contain GW SinglePoint or a list of PhotonPolarizations
        if xs_tasks[0].task.m_def.name != 'PhotonPolarization':
            gw_task = xs_tasks[0]
            xs_tasks.pop(
                0
            )  # we delete the [0] element associated with GW in case DFT+GW+PhotonPolarization workflow

        if not self.results:
            self.results = XSResults()

        task_map = {
            'dft': dft_task,
            'gw': gw_task,
        }
        self.get_electronic_structure_workflow_results(task_map)

        for xs in xs_tasks:
            if xs.m_xpath('task.results'):
                photon_results = xs.task.results
                # Adding provenance to BSE method section, in addition to the existent 'photon' provenance
                if xs.task.m_xpath('inputs[1].section'):
                    for spectra in photon_results.spectrum_polarization:
                        provenance = ElectronicStructureProvenance(
                            methodology=xs.task.inputs[1].section, label='bse'
                        )
                        spectra.m_add_sub_section(Spectra.provenance, provenance)
                self.results.m_add_sub_section(XSResults.spectra, photon_results)
