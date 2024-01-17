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
from .general import (
    SimulationWorkflow,
    SimulationWorkflowMethod,
    SimulationWorkflowResults,
    BeyondDFT,
    DFTMethod,
    ElectronicStructureOutputs,
    MagneticOutputs,
)
from .single_point import SinglePoint, SinglePointMethod, SinglePointResults
from .geometry_optimization import (
    GeometryOptimization,
    GeometryOptimizationMethod,
    GeometryOptimizationResults,
)
from .molecular_dynamics import (
    MolecularDynamics,
    MolecularDynamicsMethod,
    MolecularDynamicsResults,
)
from .phonon import Phonon, PhononMethod, PhononResults
from .equation_of_state import (
    EquationOfState,
    EquationOfStateMethod,
    EquationOfStateResults,
)
from .chemical_reaction import (
    ChemicalReaction,
    ChemicalReactionMethod,
    ChemicalReactionResults,
)
from .elastic import Elastic, ElasticMethod, ElasticResults
from .tb import (
    FirstPrinciplesPlusTB,
    FirstPrinciplesPlusTBMethod,
    FirstPrinciplesPlusTBResults,
)
from .gw import DFTPlusGW, DFTPlusGWMethod, DFTPlusGWResults
from .xs import XS, XSMethod, XSResults
from .dmft import DFTPlusTBPlusDMFT, DFTPlusTBPlusDMFTMethod, DFTPlusTBPlusDMFTResults
from .max_ent import DMFTPlusMaxEnt, DMFTPlusMaxEntMethod, DMFTPlusMaxEntResults
from .photon_polarization import (
    PhotonPolarization,
    PhotonPolarizationMethod,
    PhotonPolarizationResults,
)
from .thermodynamics import Thermodynamics, ThermodynamicsMethod, ThermodynamicsResults
