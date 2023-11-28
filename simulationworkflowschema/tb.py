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
from nomad.datamodel.metainfo.simulation.method import TB as TBMethodology
from nomad.datamodel.metainfo.simulation.calculation import BandGap, BandStructure
from .general import SimulationWorkflowResults, SimulationWorkflowMethod, SerialSimulation


class TBResults(SimulationWorkflowResults):

    band_gap_first_principles = Quantity(
        type=Reference(BandGap),
        shape=['*'],
        description='''
            Reference to the First-principles band gap.
            ''')

    band_gap_tb = Quantity(
        type=Reference(BandGap),
        shape=['*'],
        description='''
            Reference to the TB band gap.
            ''')

    band_structure_first_principles = Quantity(
        type=Reference(BandStructure),
        shape=['*'],
        description='''
        Reference to the first-principles band structure.
        ''')

    band_structure_tb = Quantity(
        type=Reference(BandStructure),
        shape=['*'],
        description='''
        Reference to the tight-Binding band structure.
        ''')


class TBMethod(SimulationWorkflowMethod):

    tb_method_ref = Quantity(
        type=Reference(TBMethodology),
        description='''
        Reference to the tight-Binding methodology.
        ''')


class TB(SerialSimulation):

    method = SubSection(sub_section=TBMethod)

    results = SubSection(sub_section=TBResults)

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if not self.method:
            self.method = TBMethod()

        if not self.results:
            self.results = TBResults()

        if len(self.tasks) != 2:
            logger.error('Expected two tasks.')
            return

        first_principles_task = self.tasks[0]
        tb_task = self.tasks[1]

        for name, section in self.results.m_def.all_quantities.items():
            calc_name = '_'.join(name.split('_')[:-1])
            if name.endswith('first_principles'):
                calc_name = '_'.join(name.split('_')[:-2])
            if calc_name in ['band_structure']:
                calc_name = f'{calc_name}_electronic'
            calc_section = []
            if 'first_principles' in name:
                calc_section = getattr(first_principles_task.outputs[-1].section, calc_name)
            elif 'tb' in name:
                calc_section = getattr(tb_task.outputs[-1].section, calc_name)
            if calc_section:
                self.results.m_set(section, calc_section)
