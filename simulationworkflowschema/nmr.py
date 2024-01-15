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
from nomad.datamodel.metainfo.simulation.method import XCFunctional, BasisSetContainer
from nomad.datamodel.metainfo.simulation.calculation import (
    BandGap,
    Dos,
    BandStructure,
    MagneticShielding,
    ElectricFieldGradient,
    SpinSpinCoupling,
    MagneticSusceptibility,
)
from .general import (
    SimulationWorkflowResults,
    SimulationWorkflowMethod,
    SerialSimulation,
)


class NMRResults(SimulationWorkflowResults):
    """Groups DFT and NMR outputs: band gaps, DOS, band structures, and magnetic outputs.
    The ResultsNormalizer takes care of adding a label 'DFT' or 'NMR' in the method
    `get_nmr_workflow_properties`.
    """

    band_gap_dft = Quantity(
        type=Reference(BandGap),
        shape=["*"],
        description="""
        Reference to the DFT band gap.
        """,
    )

    dos_dft = Quantity(
        type=Reference(Dos),
        shape=["*"],
        description="""
        Reference to the DFT density of states.
        """,
    )

    band_structure_dft = Quantity(
        type=Reference(BandStructure),
        shape=["*"],
        description="""
        Reference to the DFT band structure.
        """,
    )

    magnetic_shielding_nmr = Quantity(
        type=Reference(MagneticShielding),
        shape=["*"],
        description="""
        Reference to the NMR magnetic shielding tensors.
        """,
    )

    electric_field_gradient_nmr = Quantity(
        type=Reference(ElectricFieldGradient),
        shape=["*"],
        description="""
        Reference to the NMR electric field gradient tensors.
        """,
    )

    spin_spin_coupling_nmr = Quantity(
        type=Reference(SpinSpinCoupling),
        shape=["*"],
        description="""
        Reference to the NMR spin-spin coupling tensors.
        """,
    )

    magnetic_susceptibility_nmr = Quantity(
        type=Reference(MagneticSusceptibility),
        shape=["*"],
        description="""
        Reference to the NMR magnetic susceptibility tensors.
        """,
    )


class NMRMethod(SimulationWorkflowMethod):
    """Groups DFT and NRM input methodologies: starting XC functional, electrons
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


class NMR(SerialSimulation):
    """The NMR workflow is generated in an extra EntryArchive IF both the DFT SinglePoint
    and the NMR SinglePoint EntryArchives are present in the upload.
    """

    method = SubSection(sub_section=NMRMethod)

    results = SubSection(sub_section=NMRResults)

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if len(self.tasks) != 2:
            logger.error("Expected two tasks.")
            return

        dft_task = self.tasks[0]
        nmr_task = self.tasks[1]

        if not self.results:
            self.results = NMRResults()

        for name, section in self.results.m_def.all_quantities.items():
            calc_name = "_".join(name.split("_")[:-1])
            if calc_name in ["dos", "band_structure"]:
                calc_name = f"{calc_name}_electronic"
            calc_section = []
            if "dft" in name:
                calc_section = getattr(dft_task.outputs[-1].section, calc_name)
            elif "gw" in name:
                calc_section = getattr(nmr_task.outputs[-1].section, calc_name)
            if calc_section:
                self.results.m_set(section, calc_section)
