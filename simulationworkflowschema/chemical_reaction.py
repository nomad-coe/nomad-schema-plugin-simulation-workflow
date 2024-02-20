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
import numpy as np

from nomad.metainfo import SubSection, Section, Quantity, MEnum
from nomad.datamodel.metainfo.workflow import Task
from runschema.calculation import Calculation
from .general import (
    SimulationWorkflowMethod,
    SimulationWorkflowResults,
    SimulationWorkflow,
)


class ChemicalReactionMethod(SimulationWorkflowMethod):
    m_def = Section(validate=False)

    reaction_type = Quantity(
        type=MEnum('surface_adsorption'),
        description="""
        The type of the chemical reaction.
        """,
    )


class ChemicalReactionResults(SimulationWorkflowResults):
    reaction_energy = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Calculated value of the reaction energy, E_reaction= E_products - E_reactants
        """,
    )

    activation_energy = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Calculated value of the activation energy, E_activation = E_transitions - E_reactants
        """,
    )


class ChemicalReaction(SimulationWorkflow):
    method = SubSection(sub_section=ChemicalReactionMethod)

    results = SubSection(sub_section=ChemicalReactionResults)

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if not self.method:
            self.method = ChemicalReactionMethod()

        if not self.results:
            self.results = ChemicalReactionResults()

        # identify if input is rectant or product
        reactants, products, transitions = [], [], []
        for input in self.inputs:
            calculation = input.section
            if calculation.m_def.section_cls != Calculation or not input.name:
                continue
            if calculation.m_xpath('energy.total.value') is None:
                logger.error(
                    'Calculation energy required to calculate adsorption energy.'
                )
                reactants, products, transitions = [], [], []
                break
            # TODO resolve reagent type automatically
            if 'product' in input.name.lower():
                products.append(calculation)
            elif 'transition' in input.name.lower():
                transitions.append(calculation)
            elif 'reactant' in input.name.lower():
                reactants.append(calculation)

        def get_lattices(calculations):
            lattices = []
            for calculation in calculations:
                dimensionality = calculation.m_parent.m_parent.m_xpath(
                    'results.material.dimensionality'
                )
                if not dimensionality or dimensionality == '0D':
                    continue
                lattice = calculation.m_xpath('system_ref.atoms.lattice_vectors')
                if lattice is not None:
                    lattices.append(lattice)
            return lattices

        # check compatibiliy of cell sizes
        reactant_lattices = get_lattices(reactants)
        for reference_lattice in reactant_lattices:
            for lattice in get_lattices(products):
                array_equal = np.isclose(reference_lattice, lattice, atol=0, rtol=1e-6)
                if not array_equal[
                    np.where((reference_lattice != 0) & (lattice != 0))
                ].all():
                    logger.error('Reactant and product lattices do not match.')
                    reactants, products = [], []
                    break
            for lattice in get_lattices(transitions):
                array_equal = np.isclose(reference_lattice, lattice, atol=0, rtol=1e-6)
                if not array_equal[
                    np.where((reference_lattice != 0) & (lattice != 0))
                ].all():
                    logger.error('Reactant and transition state lattices do not match.')
                    transitions = []
                    break

        def get_labels(calculations):
            atom_labels = []
            for calculation in calculations:
                labels = calculation.m_xpath('system_ref.atoms.labels')
                if labels:
                    atom_labels.extend(labels)
            return sorted(atom_labels)

        # check consistency of composition, i.e. sum of species in reactants should equal
        # products and transition state
        reactants_labels = get_labels(reactants)
        if reactants_labels != get_labels(products):
            logger.error('Inconsistent composition of reactants and products.')
            reactants, products = [], []
        if transitions and reactants_labels != get_labels(transitions):
            logger.error('Inconsistent composition of reactants and transition states.')
            transitions = []

        energy_reactants = np.sum(
            [calc.energy.total.value.magnitude for calc in reactants]
        )
        energy_products = np.sum(
            [calc.energy.total.value.magnitude for calc in products]
        )
        energy_transitions = np.sum(
            [calc.energy.total.value.magnitude for calc in transitions]
        )

        if reactants and products:
            self.results.reaction_energy = energy_products - energy_reactants

        if reactants and transitions:
            self.results.activation_energy = energy_transitions - energy_reactants

        # create task for calculating reaction energy
        self.tasks.append(
            Task(
                name='Calculation of reaction energy.',
                inputs=self.inputs,
                outputs=self.outputs,
            )
        )
