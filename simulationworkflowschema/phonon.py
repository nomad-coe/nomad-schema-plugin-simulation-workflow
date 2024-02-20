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

from nomad.metainfo import SubSection, Quantity, Reference
from nomad.datamodel.metainfo.workflow import Link
from runschema.calculation import Dos, BandStructure
from .general import (
    ParallelSimulation,
    SimulationWorkflowMethod,
    SimulationWorkflowResults,
    WORKFLOW_METHOD_NAME,
    WORKFLOW_RESULTS_NAME,
)
from .thermodynamics import ThermodynamicsResults


class PhononMethod(SimulationWorkflowMethod):
    force_calculator = Quantity(
        type=str,
        shape=[],
        description="""
        Name of the program used to calculate the forces.
        """,
    )

    mesh_density = Quantity(
        type=np.float64,
        shape=[],
        unit='1 / meter ** 3',
        description="""
        Density of the k-mesh for sampling.
        """,
    )

    random_displacements = Quantity(
        type=bool,
        shape=[],
        description="""
        Identifies if displacements are made randomly.
        """,
    )

    with_non_analytic_correction = Quantity(
        type=bool,
        shape=[],
        description="""
        Identifies if non-analytical term corrections are applied to dynamical matrix.
        """,
    )

    with_grueneisen_parameters = Quantity(
        type=bool,
        shape=[],
        description="""
        Identifies if Grueneisen parameters are calculated.
        """,
    )


class PhononResults(ThermodynamicsResults):
    n_imaginary_frequencies = Quantity(
        type=int,
        shape=[],
        description="""
        Number of modes with imaginary frequencies.
        """,
    )

    n_bands = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of phonon bands.
        """,
    )

    n_qpoints = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of q points for which phonon properties are evaluated.
        """,
    )

    qpoints = Quantity(
        type=np.float64,
        shape=['n_qpoints', 3],
        description="""
        Value of the qpoints.
        """,
    )

    group_velocity = Quantity(
        type=np.float64,
        shape=['n_qpoints', 'n_bands', 3],
        unit='meter / second',
        description="""
        Calculated value of the group velocity at each qpoint.
        """,
    )

    n_displacements = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of independent displacements.
        """,
    )

    n_atoms = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of atoms in the simulation cell.
        """,
    )

    displacements = Quantity(
        type=np.float64,
        shape=['n_displacements', 'n_atoms', 3],
        unit='meter',
        description="""
        Value of the displacements applied to each atom in the simulation cell.
        """,
    )

    dos = Quantity(
        type=Reference(Dos),
        shape=['n_data'],
        description="""
        Reference to the electronic density of states data.
        """,
    )

    band_structure = Quantity(
        type=Reference(BandStructure),
        shape=['n_data'],
        description="""
        Reference to the electronic band structure data.
        """,
    )


class Phonon(ParallelSimulation):
    method = SubSection(sub_section=PhononMethod)

    results = SubSection(sub_section=PhononResults)

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if not self._calculations or not self._calculations[0].band_structure_phonon:
            return

        if not self.method:
            self.method = PhononMethod()
            self.inputs.append(Link(name=WORKFLOW_METHOD_NAME, section=self.method))

        if not self.results:
            self.results = PhononResults()
            self.outputs.append(Link(name=WORKFLOW_RESULTS_NAME, section=self.results))

        ThermodynamicsResults.normalize(self.results, archive, logger)

        last_calc = self._calculations[-1]

        if not self.results.n_imaginary_frequencies:
            n_imaginary = 0
            for band_segment in last_calc.band_structure_phonon[-1].segment:
                freq = band_segment.energies.magnitude
                n_imaginary += np.count_nonzero(np.array(freq) < 0)
            self.results.n_imaginary_frequencies = n_imaginary

        SimulationWorkflowResults.normalize(self.results, archive, logger)

        if not self.results.dos:
            self.results.dos = last_calc.dos_phonon

        if not self.results.band_structure:
            self.results.band_structure = last_calc.band_structure_phonon
