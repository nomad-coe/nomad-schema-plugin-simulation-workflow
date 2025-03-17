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
from typing import Optional

from nomad.datamodel.data import ArchiveSection
from nomad.metainfo import SubSection, Section, Quantity, Reference, derived
from nomad.datamodel.metainfo.workflow import Link
from runschema.system import System
from .general import (
    SimulationWorkflowResults,
    SimulationWorkflowMethod,
    SerialSimulation,
    WORKFLOW_METHOD_NAME,
    WORKFLOW_RESULTS_NAME,
)


class Decomposition(ArchiveSection):
    """
    Section containing information about the system to which an unstable compound will
    decompose to.
    """

    m_def = Section(validate=False)

    fraction = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Amount of the resulting system.
        """,
    )

    system_ref = Quantity(
        type=Reference(System.m_def),
        shape=[],
        description="""
        Reference to the resulting system.
        """,
    )

    formula = Quantity(
        type=str,
        shape=[],
        description="""
        Chemical formula of the resulting system.
        """,
    )


class Stability(ArchiveSection):
    """
    Section containing information regarding the stability of the system.
    """

    m_def = Section(validate=False)

    n_references = Quantity(
        type=int,
        shape=[],
        description="""
        Number of reference systems.
        """,
    )

    systems_ref = Quantity(
        type=Reference(System.m_def),
        shape=['n_references'],
        description="""
        References to the reference systems.
        """,
    )

    formation_energy = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        Calculated value of the formation energy of the compound.
        """,
    )

    delta_formation_energy = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        Energy with respect to the convex hull.
        """,
    )

    n_references = Quantity(
        type=int,
        shape=[],
        description="""
        Number of reference systems.
        """,
    )

    is_stable = Quantity(
        type=bool,
        shape=[],
        description="""
        Indicates if a compound is stable.
        """,
    )

    decomposition = SubSection(sub_section=Decomposition.m_def, repeats=True)


class ThermodynamicsResults(SimulationWorkflowResults):
    n_values = Quantity(
        type=int,
        shape=[],
        description="""
        Number of thermodynamics property evaluations.
        """,
    )

    temperature = Quantity(
        type=np.float64,
        shape=['n_values'],
        unit='kelvin',
        description="""
        Specifies the temperatures at which properties such as the Helmholtz free energy
        are calculated.
        """,
    )

    pressure = Quantity(
        type=np.float64,
        shape=['n_values'],
        unit='pascal',
        description="""
        Array containing the values of the pressure (one third of the trace of the stress
        tensor) corresponding to each property evaluation.
        """,
    )

    helmholtz_free_energy = Quantity(
        type=np.float64,
        shape=['n_values'],
        unit='joule',
        description="""
        Helmholtz free energy per unit cell at constant volume.
        """,
    )

    heat_capacity_c_p = Quantity(
        type=np.float64,
        shape=['n_values'],
        unit='joule / kelvin',
        description="""
        Heat capacity per cell unit at constant pressure.
        """,
    )

    heat_capacity_c_v = Quantity(
        type=np.float64,
        shape=['n_values'],
        unit='joule / kelvin',
        description="""
        Heat capacity per cell unit at constant volume.
        """,
    )

    @derived(
        type=np.float64,
        shape=['n_values'],
        unit='joule / (kelvin * kilogram)',
        description="""
        Specific heat capacity at constant volume.
        """,
        cached=True,
    )
    def heat_capacity_c_v_specific(self) -> Optional[np.ndarray]:
        """Returns the specific heat capacity by dividing the heat capacity per
        cell with the mass of the atoms in the cell.
        """
        import nomad.atomutils

        workflow = self.m_parent
        if not workflow._systems or not workflow._systems[0].atoms:
            return None
        atomic_numbers = workflow._systems[0].atoms.species
        mass_per_unit_cell = nomad.atomutils.get_summed_atomic_mass(atomic_numbers)
        heat_capacity = self.heat_capacity_c_v
        specific_heat_capacity = heat_capacity / mass_per_unit_cell

        return specific_heat_capacity.magnitude

    vibrational_free_energy_at_constant_volume = Quantity(
        type=np.float64,
        shape=['n_values'],
        unit='joule',
        description="""
        Holds the vibrational free energy per cell unit at constant volume.
        """,
    )

    @derived(
        type=np.float64,
        shape=['n_values'],
        unit='joule / kilogram',
        description="""
        Stores the specific vibrational free energy at constant volume.
        """,
        cached=True,
    )
    def vibrational_free_energy_at_constant_volume_specific(
        self,
    ) -> Optional[np.ndarray]:
        import nomad.atomutils

        workflow = self.m_parent
        if not workflow._systems or not workflow._systems[0].atoms:
            return None
        atomic_numbers = workflow._systems[0].atoms.species
        mass_per_unit_cell = nomad.atomutils.get_summed_atomic_mass(atomic_numbers)
        free_energy = self.vibrational_free_energy_at_constant_volume
        specific_free_energy = free_energy / mass_per_unit_cell

        return specific_free_energy.magnitude

    vibrational_free_energy = Quantity(
        type=np.float64,
        shape=['n_values'],
        unit='joule',
        description="""
        Calculated value of the vibrational free energy, F_vib.
        """,
    )

    vibrational_internal_energy = Quantity(
        type=np.float64,
        shape=['n_values'],
        unit='joule',
        description="""
        Calculated value of the vibrational internal energy, U_vib.
        """,
    )

    vibrational_entropy = Quantity(
        type=np.float64,
        shape=['n_values'],
        unit='joule / kelvin',
        description="""
        Calculated value of the vibrational entropy, S.
        """,
    )

    gibbs_free_energy = Quantity(
        type=np.float64,
        shape=['n_values'],
        unit='joule',
        description="""
        Calculated value of the Gibbs free energy, G.
        """,
    )

    entropy = Quantity(
        type=np.float64,
        shape=['n_values'],
        unit='joule / kelvin',
        description="""
        Calculated value of the entropy.
        """,
    )

    enthalpy = Quantity(
        type=np.float64,
        shape=['n_values'],
        unit='joule',
        description="""
        Calculated value of enthalpy.
        """,
    )

    internal_energy = Quantity(
        type=np.float64,
        shape=['n_values'],
        unit='joule',
        description="""
        Calculated value of the internal energy, U.
        """,
    )

    stability = SubSection(sub_section=Stability.m_def, repeats=False)

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        try:
            calculations = archive.run[-1].calculation
        except Exception:
            calculations = []

        def set_thermo_property(name):
            values = []
            quantity = None
            for calc in calculations:
                if hasattr(calc, name):
                    try:
                        quantity = calc[name]
                        values.append(
                            quantity.magnitude
                            if hasattr(quantity, 'magnitude')
                            else quantity
                        )
                        continue
                    except Exception:
                        pass
                # TODO section thermodynamics should be removed
                for thermo in calc.thermodynamics:
                    try:
                        quantity = thermo[name]
                        values.append(
                            quantity.magnitude
                            if hasattr(quantity, 'magnitude')
                            else quantity
                        )
                    except Exception:
                        pass
            if len(values) == 0:
                return

            unit = quantity.units if hasattr(quantity, 'units') else 1.0
            setattr(self, name, np.array(values) * unit)

        if self.temperature is None:
            set_thermo_property('temperature')

        if self.helmholtz_free_energy is None:
            set_thermo_property('helmholtz_free_energy')

        if self.vibrational_free_energy_at_constant_volume is None:
            set_thermo_property('vibrational_free_energy_at_constant_volume')

        if self.heat_capacity_c_v is None:
            set_thermo_property('heat_capacity_c_v')


class ThermodynamicsMethod(SimulationWorkflowMethod):
    pass


class Thermodynamics(SerialSimulation):
    method = SubSection(sub_section=ThermodynamicsMethod)

    results = SubSection(sub_section=ThermodynamicsResults)

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if not self.method:
            self.method = ThermodynamicsMethod()
            self.inputs.append(Link(name=WORKFLOW_METHOD_NAME, section=self.method))

        if not self.results:
            self.results = ThermodynamicsResults()
            self.outputs.append(Link(name=WORKFLOW_RESULTS_NAME, section=self.results))
