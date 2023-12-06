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
    XCFunctional, BasisSetContainer, GW as GWMethodology
)
from runschema.calculation import BandGap, Dos, BandStructure
from .general import SimulationWorkflowResults, SimulationWorkflowMethod, SerialSimulation


class GWResults(SimulationWorkflowResults):
    '''Groups DFT and GW outputs: band gaps, DOS, band structures. The ResultsNormalizer
    takes care of adding a label 'DFT' or 'GW' in the method `get_gw_workflow_properties`.
    '''

    band_gap_dft = Quantity(
        type=Reference(BandGap),
        shape=['*'],
        description='''
        Reference to the DFT band gap.
        ''')

    band_gap_gw = Quantity(
        type=Reference(BandGap),
        shape=['*'],
        description='''
        Reference to the GW band gap.
        ''')

    dos_dft = Quantity(
        type=Reference(Dos),
        shape=['*'],
        description='''
        Reference to the DFT density of states.
        ''')

    dos_gw = Quantity(
        type=Reference(Dos),
        shape=['*'],
        description='''
        Reference to the GW density of states.
        ''')

    band_structure_dft = Quantity(
        type=Reference(BandStructure),
        shape=['*'],
        description='''
        Reference to the DFT band structure.
        ''')

    band_structure_gw = Quantity(
        type=Reference(BandStructure),
        shape=['*'],
        description='''
        Reference to the GW band structure.
        ''')


class GWMethod(SimulationWorkflowMethod):
    '''Groups DFT and GW input methodologies: starting XC functional, electrons
    representation (basis set), GW method reference.
    '''

    starting_point = Quantity(
        type=Reference(XCFunctional),
        description='''
        Reference to the starting point (XC functional or HF) used.
        ''')

    electrons_representation = Quantity(
        type=Reference(BasisSetContainer),
        description='''
        Reference to the basis set used.
        ''')

    gw_method_ref = Quantity(
        type=Reference(GWMethodology),
        description='''
        Reference to the GW methodology.
        ''')


class GW(SerialSimulation):
    '''The GW workflow is generated in an extra EntryArchive IF both the DFT SinglePoint
    and the GW SinglePoint EntryArchives are present in the upload.
    '''

    method = SubSection(sub_section=GWMethod)

    results = SubSection(sub_section=GWResults)

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if len(self.tasks) != 2:
            logger.error('Expected two tasks.')
            return

        dft_task = self.tasks[0]
        gw_task = self.tasks[1]

        if not self.results:
            self.results = GWResults()

        for name, section in self.results.m_def.all_quantities.items():
            calc_name = '_'.join(name.split('_')[:-1])
            if calc_name in ['dos', 'band_structure']:
                calc_name = f'{calc_name}_electronic'
            calc_section = []
            if 'dft' in name:
                calc_section = getattr(dft_task.outputs[-1].section, calc_name)
            elif 'gw' in name:
                calc_section = getattr(gw_task.outputs[-1].section, calc_name)
            if calc_section:
                self.results.m_set(section, calc_section)
