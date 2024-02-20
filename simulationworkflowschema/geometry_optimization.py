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
from ase import Atoms

from nomad.metainfo import SubSection, Quantity, MEnum
from nomad.datamodel.metainfo.workflow import Link
from runschema.calculation import EnergyEntry

from .general import (
    SimulationWorkflowMethod,
    SimulationWorkflowResults,
    SerialSimulation,
    WORKFLOW_METHOD_NAME,
    WORKFLOW_RESULTS_NAME,
    resolve_difference,
)


class GeometryOptimizationMethod(SimulationWorkflowMethod):
    type = Quantity(
        type=MEnum('static', 'atomic', 'cell_shape', 'cell_volume'),
        shape=[],
        description="""
        The type of geometry optimization, which denotes what is being optimized.

        Allowed values are:

        | Type                   | Description                               |

        | ---------------------- | ----------------------------------------- |

        | `"static"`             | no optimization |

        | `"atomic"`             | the atomic coordinates alone are updated |

        | `"cell_volume"`         | `"atomic"` + cell lattice paramters are updated isotropically |

        | `"cell_shape"`        | `"cell_volume"` but without the isotropic constraint: all cell parameters are updated |

        """,
    )

    method = Quantity(
        type=str,
        shape=[],
        description="""
        The method used for geometry optimization. Some known possible values are:
        `"steepest_descent"`, `"conjugant_gradient"`, `"low_memory_broyden_fletcher_goldfarb_shanno"`.
        """,
    )

    convergence_tolerance_energy_difference = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        The input energy difference tolerance criterion.
        """,
    )

    convergence_tolerance_force_maximum = Quantity(
        type=np.float64,
        shape=[],
        unit='newton',
        description="""
        The input maximum net force tolerance criterion.
        """,
    )

    convergence_tolerance_stress_maximum = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        The input maximum stress tolerance criterion.
        """,
    )

    convergence_tolerance_displacement_maximum = Quantity(
        type=np.float64,
        shape=[],
        unit='meter',
        description="""
        The input maximum displacement tolerance criterion.
        """,
    )

    optimization_steps_maximum = Quantity(
        type=int,
        shape=[],
        description="""
        Maximum number of optimization steps.
        """,
    )

    save_frequency = Quantity(
        type=int,
        shape=[],
        description="""
        The number of optimization steps between saving the calculation.
        """,
    )


class GeometryOptimizationResults(SimulationWorkflowResults):
    optimization_steps = Quantity(
        type=int,
        shape=[],
        description="""
        Number of saved optimization steps.
        """,
    )

    energies = Quantity(
        type=np.float64,
        unit='joule',
        shape=['optimization_steps'],
        description="""
        List of energy_total values gathered from the single configuration
        calculations that are a part of the optimization trajectory.
        """,
    )

    steps = Quantity(
        type=np.int32,
        shape=['optimization_steps'],
        description="""
        The step index corresponding to each saved configuration.
        """,
    )

    final_energy_difference = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        The difference in the energy_total between the last two steps during
        optimization.
        """,
    )

    final_force_maximum = Quantity(
        type=np.float64,
        shape=[],
        unit='newton',
        description="""
        The maximum net force in the last optimization step.
        """,
    )

    final_displacement_maximum = Quantity(
        type=np.float64,
        shape=[],
        unit='meter',
        description="""
        The maximum displacement in the last optimization step with respect to previous.
        """,
    )

    is_converged_geometry = Quantity(
        type=bool,
        shape=[],
        description="""
        Indicates if the geometry convergence criteria were fulfilled.
        """,
    )


class GeometryOptimization(SerialSimulation):
    method = SubSection(sub_section=GeometryOptimizationMethod)

    results = SubSection(sub_section=GeometryOptimizationResults)

    def _get_geometry_optimization_type(self):
        if not self._systems:
            return

        def compare_cell(cell1, cell2):
            if (cell1 == cell2).all():
                return None
            else:
                cell1_normed = cell1 / np.linalg.norm(cell1)
                cell2_normed = cell2 / np.linalg.norm(cell2)
                if (cell1_normed == cell2_normed).all():
                    return 'cell_volume'
                else:
                    return 'cell_shape'

        if len(self._systems) < 2:
            return 'static'

        else:
            if self._systems[0].atoms is None or self._systems[-1].atoms is None:
                return 'static'

            cell_init = self._systems[0].atoms.lattice_vectors
            cell_final = self._systems[-1].atoms.lattice_vectors
            if cell_init is None or cell_final is None:
                return 'static'

            cell_relaxation = compare_cell(cell_init.magnitude, cell_final.magnitude)

            if cell_relaxation is not None:
                return cell_relaxation

            atom_pos_init = self._systems[0].atoms.positions
            atom_pos_final = self._systems[-1].atoms.positions
            if atom_pos_init is None or atom_pos_final is None:
                return 'static'

            if (atom_pos_init.magnitude == atom_pos_final.magnitude).all():
                return 'static'

            return 'atomic'

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if not self.method:
            self.method = GeometryOptimizationMethod()
            self.inputs.append(Link(name=WORKFLOW_METHOD_NAME, section=self.method))

        if not self.results:
            self.results = GeometryOptimizationResults()
            self.outputs.append(Link(name=WORKFLOW_RESULTS_NAME, section=self.results))

        if not self.method.type and self._systems:
            self.method.type = self._get_geometry_optimization_type()

        if not self.results.optimization_steps:
            self.results.optimization_steps = len(self._calculations)

        energies = []
        invalid = False
        for calc in self._calculations:
            try:
                energy = calc.energy.total.value
            except (IndexError, AttributeError):
                invalid = True
                break
            if energy is None:
                invalid = True
                break
            energies.append(energy.magnitude)
        if invalid:
            logger.warning(
                'Energy not reported for an calculation that is part of a geometry optimization'
            )
        if energies:
            self.results.energies = energies * EnergyEntry.value.unit

        energy_difference = resolve_difference(energies)
        if not self.results.final_energy_difference and energy_difference is not None:
            self.results.final_energy_difference = (
                energy_difference * EnergyEntry.value.unit
            )

        if not self.results.final_force_maximum:
            if len(self._calculations) > 0:
                if (
                    self._calculations[-1].forces is not None
                    and self._calculations[-1].forces.total is not None
                ):
                    forces = self._calculations[-1].forces.total.value
                    if forces is not None:
                        max_force = np.max(np.linalg.norm(forces.magnitude, axis=1))
                        self.results.final_force_maximum = max_force * forces.units

        if not self.results.final_displacement_maximum and self._systems:

            def get_atoms_positions(index):
                system = self._systems[index]
                if (
                    not system.atoms
                    or system.atoms.positions is None
                    or system.atoms.lattice_vectors is None
                ):
                    return
                try:
                    atoms = Atoms(
                        positions=system.atoms.positions.magnitude,
                        cell=system.atoms.lattice_vectors.magnitude,
                        pbc=system.atoms.periodic,
                    )
                    atoms.wrap()
                    return atoms.get_positions()
                except Exception:
                    return

            n_systems = len(self._systems)
            a_pos = get_atoms_positions(n_systems - 1)
            if a_pos is not None:
                for i in range(n_systems - 2, -1, -1):
                    b_pos = get_atoms_positions(i)
                    if b_pos is None:
                        continue
                    displacement_maximum = np.max(np.abs(a_pos - b_pos))
                    if displacement_maximum > 0:
                        self.results.final_displacement_maximum = displacement_maximum
                        break

        if not self.results.is_converged_geometry:
            # we can have several criteria for convergence: energy, force, displacement
            criteria = []
            try:
                criteria.append(
                    self.results.final_energy_difference
                    <= self.method.convergence_tolerance_energy_difference
                )
            except Exception:
                pass

            try:
                criteria.append(
                    self.results.final_force_maximum
                    <= self.method.convergence_tolerance_force_maximum
                )
            except Exception:
                pass

            try:
                criteria.append(
                    self.results.final_displacement_maximum
                    <= self.method.convergence_tolerance_displacement_maximum
                )
            except Exception:
                pass

            # converged when either criterion is met
            if criteria:
                self.results.is_converged_geometry = True in criteria

        SimulationWorkflowResults.normalize(self.results, archive, logger)
