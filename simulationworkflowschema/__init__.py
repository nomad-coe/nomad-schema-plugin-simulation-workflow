#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
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
import importlib

from nomad.config.models.plugins import SchemaPackageEntryPoint


__all__ = [
    'SimulationWorkflow',
    'SimulationWorkflowMethod',
    'SimulationWorkflowResults',
    'ParallelSimulation',
    'SerialSimulation',
    'BeyondDFT',
    'DFTMethod',
    'ElectronicStructureOutputs',
    'MagneticOutputs',
    'SinglePoint',
    'SinglePointMethod',
    'SinglePointResults',
    'GeometryOptimization',
    'GeometryOptimizationMethod',
    'GeometryOptimizationResults',
    'MolecularDynamics',
    'MolecularDynamicsMethod',
    'MolecularDynamicsResults',
    'Phonon',
    'PhononMethod',
    'PhononResults',
    'EquationOfState',
    'EquationOfStateMethod',
    'EquationOfStateResults',
    'ChemicalReaction',
    'ChemicalReactionMethod',
    'ChemicalReactionResults',
    'Elastic',
    'ElasticMethod',
    'ElasticResults',
    'FirstPrinciplesPlusTB',
    'FirstPrinciplesPlusTBMethod',
    'FirstPrinciplesPlusTBResults',
    'DFTPlusGW',
    'DFTPlusGWMethod',
    'DFTPlusGWResults',
    'XS',
    'XSMethod',
    'XSResults',
    'DFTPlusTBPlusDMFT',
    'DFTPlusTBPlusDMFTMethod',
    'DFTPlusTBPlusDMFTResults',
    'DMFTPlusMaxEnt',
    'DMFTPlusMaxEntMethod',
    'DMFTPlusMaxEntResults',
    'PhotonPolarization',
    'PhotonPolarizationMethod',
    'PhotonPolarizationResults',
    'Thermodynamics',
    'ThermodynamicsMethod',
    'ThermodynamicsResults',
]


def load_modules():
    sub_modules = [
        'general',
        'single_point',
        'geometry_optimization',
        'molecular_dynamics',
        'phonon',
        'equation_of_state',
        'chemical_reaction',
        'elastic',
        'tb',
        'gw',
        'xs',
        'dmft',
        'max_ent',
        'photon_polarization',
        'thermodynamics',
    ]
    import simulationworkflowschema

    for name in sub_modules:
        sub_module = importlib.import_module(f'simulationworkflowschema.{name}')
        for method in sub_module.__dict__:
            if method in __all__:
                setattr(simulationworkflowschema, method, sub_module.__dict__[method])


class SimulationWorkflowSchemaEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        load_modules()

        from .general import m_package

        return m_package


simulationworkflow_schema_entry_point = SimulationWorkflowSchemaEntryPoint(
    name='SimulationWorkflowSchema',
    description='Schema for the nomad simulation workflows.',
)
