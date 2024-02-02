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
    XCFunctional,
    BasisSetContainer,
    TB as TBMethodology,
    DMFT as DMFTMethodology,
)
from runschema.calculation import BandGap, Dos, BandStructure, GreensFunctions
from .general import (
    SimulationWorkflowResults,
    SimulationWorkflowMethod,
    SerialSimulation,
)


class DMFTResults(SimulationWorkflowResults):
    """Groups DFT, TB and DMFT outputs: band gaps (all), DOS (DFT, TB), band
    structures (DFT, TB), Greens functions (DMFT). The ResultsNormalizer takes care
    of adding a label 'DFT', 'PROJECTION, or 'DMFT' in the method `get_dmft_workflow_properties`.
    """

    band_gap_dft = Quantity(
        type=Reference(BandGap),
        shape=["*"],
        description="""
        DFT band gap.
        """,
    )

    band_gap_tb = Quantity(
        type=Reference(BandGap),
        shape=["*"],
        description="""
        TB band gap.
        """,
    )

    band_gap_dmft = Quantity(
        type=Reference(BandGap),
        shape=["*"],
        description="""
        DMFT band gap.
        """,
    )

    band_structure_dft = Quantity(
        type=Reference(BandStructure),
        shape=["*"],
        description="""
        Ref to the DFT band structure.
        """,
    )

    dos_dft = Quantity(
        type=Reference(Dos),
        shape=["*"],
        description="""
        Ref to the DFT density of states.
        """,
    )

    band_structure_tb = Quantity(
        type=Reference(BandStructure),
        shape=["*"],
        description="""
        Ref to the TB band structure.
        """,
    )

    dos_tb = Quantity(
        type=Reference(Dos),
        shape=["*"],
        description="""
        Ref to the TB density of states.
        """,
    )

    greens_functions_dmft = Quantity(
        type=Reference(GreensFunctions),
        shape=["*"],
        description="""
        Ref to the DMFT Greens functions.
        """,
    )


class DMFTMethod(SimulationWorkflowMethod):
    """Groups DFT, TB and DMFT input methodologies: starting XC functional, electrons
    representation (basis set), TB method reference, DMFT method reference.
    """

    starting_point = Quantity(
        type=Reference(XCFunctional),
        description="""
        Starting point (XC functional or HF) used.
        """,
    )

    electrons_representation = Quantity(
        type=Reference(BasisSetContainer),
        description="""
        Basis set used.
        """,
    )

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


class DMFT(SerialSimulation):
    """The DMFT workflow is generated in an extra EntryArchive IF both the TB SinglePoint
    and the DMFT SinglePoint EntryArchives are present in the upload.
    """

    # TODO extend to reference a DFT SinglePoint.

    method = SubSection(sub_section=DMFTMethod)

    results = SubSection(sub_section=DMFTResults)

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if len(self.tasks) != 2:
            logger.error("Expected two tasks: TB and DMFT SinglePoint tasks")
            return

        proj_task = self.tasks[0]
        dmft_task = self.tasks[1]

        if not self.results:
            self.results = DMFTResults()

        for name, section in self.results.m_def.all_quantities.items():
            calc_name = "_".join(name.split("_")[:-1])
            if calc_name in ["dos", "band_structure"]:
                calc_name = f"{calc_name}_electronic"
            calc_section = []
            if "tb" in name:
                calc_section = getattr(proj_task.outputs[-1].section, calc_name)
            elif "dmft" in name:
                calc_section = getattr(dmft_task.outputs[-1].section, calc_name)
            if calc_section:
                self.results.m_set(section, calc_section)
