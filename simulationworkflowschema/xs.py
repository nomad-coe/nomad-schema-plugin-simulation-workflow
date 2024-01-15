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
from nomad.datamodel.metainfo.simulation.calculation import (
    Spectra,
    ElectronicStructureProvenance,
)
from .general import DFTOutputs, GWOutputs, SimulationWorkflowMethod, SerialSimulation
from .photon_polarization import PhotonPolarizationResults


class XSResults(DFTOutputs, GWOutputs):
    """
    Groups DFT, GW and PhotonPolarization outputs: band gaps (DFT, GW), DOS (DFT, GW),
    band structures (DFT, GW), spectra (PhotonPolarization). The ResultsNormalizer takes
    care of adding a label 'DFT' or 'GW' in the method `get_xs_workflow_properties`.
    """

    spectra = SubSection(sub_section=PhotonPolarizationResults, repeats=True)


class XSMethod(SimulationWorkflowMethod):
    pass


class XS(SerialSimulation):
    """
    The XS workflow is generated in an extra EntryArchive IF both the DFT SinglePoint
    and the PhotonPolarization EntryArchives are present in the upload.
    """

    # TODO extend to reference a GW SinglePoint.

    method = SubSection(sub_section=XSMethod)

    results = SubSection(sub_section=XSResults)

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if len(self.tasks) < 2:
            logger.error(
                "Expected more than one task: DFT+PhotonPolarization or DFT+GW+PhotonPolarization."
            )
            return

        dft_task = self.tasks[0]
        xs_tasks = [self.tasks[i] for i in range(1, len(self.tasks))]
        gw_task = None
        # Check if the xs_tasks contain GW SinglePoint or a list of PhotonPolarizations
        if xs_tasks[0].task.m_def.name != "PhotonPolarization":
            gw_task = xs_tasks[0]
            xs_tasks.pop(
                0
            )  # we delete the [0] element associated with GW in case DFT+GW+PhotonPolarization workflow

        if not self.results:
            self.results = XSResults()

        for name, section in self.results.m_def.all_quantities.items():
            calc_name = "_".join(name.split("_")[:-1])
            if calc_name in ["dos", "band_structure"]:
                calc_name = f"{calc_name}_electronic"
            calc_section = []
            if "dft" in name:
                calc_section = getattr(dft_task.outputs[-1].section, calc_name)
            elif "gw" in name and gw_task:
                calc_section = getattr(gw_task.outputs[-1].section, calc_name)
            elif name == "spectra":
                pass
            if calc_section:
                self.results.m_set(section, calc_section)
        for xs in xs_tasks:
            if xs.m_xpath("task.results"):
                photon_results = xs.task.results
                # Adding provenance to BSE method section, in addition to the existent 'photon' provenance
                if xs.task.m_xpath("inputs[1].section"):
                    for spectra in photon_results.spectrum_polarization:
                        provenance = ElectronicStructureProvenance(
                            methodology=xs.task.inputs[1].section, label="bse"
                        )
                        spectra.m_add_sub_section(Spectra.provenance, provenance)
                self.results.m_add_sub_section(XSResults.spectra, photon_results)
