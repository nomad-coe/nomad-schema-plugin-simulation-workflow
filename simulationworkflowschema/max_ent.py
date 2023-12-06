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
from runschema.calculation import BandGap, Dos, GreensFunctions
from .general import SimulationWorkflowResults, SimulationWorkflowMethod, SerialSimulation


class MaxEntResults(SimulationWorkflowResults):
    '''Groups DMFT and MaxEnt outputs: greens functions (DMFT, MaxEnt), band gaps (MaxEnt),
    DOS (MaxEnt), band structures (MaxEnt). The ResultsNormalizer takes care of adding a
    label 'DMFT' or 'MaxEnt' in the method `get_maxent_workflow_properties`.
    '''

    greens_functions_dmft = Quantity(
        type=Reference(GreensFunctions),
        shape=['*'],
        description='''
        Ref to the DMFT Greens functions.
        ''')

    band_gap_maxent = Quantity(
        type=Reference(BandGap),
        shape=['*'],
        description='''
        MaxEnt band gap.
        ''')

    dos_maxent = Quantity(
        type=Reference(Dos),
        shape=['*'],
        description='''
        Ref to the MaxEnt density of states (also called spectral function).
        ''')

    greens_functions_maxent = Quantity(
        type=Reference(GreensFunctions),
        shape=['*'],
        description='''
        Ref to the MaxEnt Greens functions.
        ''')


class MaxEntMethod(SimulationWorkflowMethod):
    '''Groups DMFT and MaxEnt input methodologies: DMFT method references, MaxEnt method reference.
    '''

    dmft_method_ref = Quantity(
        type=Reference(DMFTMethodology),
        description='''
        DMFT methodology reference.
        ''')

    # TODO define MaxEnt metainfo in Method
    maxent_method_ref = Quantity(
        type=Reference(Method),
        description='''
        MaxEnt methodology reference.
        ''')


class MaxEnt(SerialSimulation):
    '''The MaxEnt (Maximum Entropy) workflow is generated in an extra EntryArchive IF both
    the DMFT SinglePoint and the MaxEnt SinglePoint EntryArchives are present in the upload.
    '''

    method = SubSection(sub_section=MaxEntMethod)

    results = SubSection(sub_section=MaxEntResults)

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if len(self.tasks) != 2:
            logger.error('Expected two tasks: DMFT and MaxEnt SinglePoint tasks')
            return

        dmft_task = self.tasks[0]
        maxent_task = self.tasks[1]

        if not self.results:
            self.results = MaxEntResults()

        for name, section in self.results.m_def.all_quantities.items():
            calc_name = '_'.join(name.split('_')[:-1])
            if calc_name in ['dos', 'band_structure']:
                calc_name = f'{calc_name}_electronic'
            calc_section = []
            if 'dmft' in name:
                calc_section = getattr(dmft_task.outputs[-1].section, calc_name)
            elif 'maxent' in name:
                calc_section = getattr(maxent_task.outputs[-1].section, calc_name)
            if calc_section:
                self.results.m_set(section, calc_section)
