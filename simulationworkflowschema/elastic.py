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

from nomad.datamodel.data import ArchiveSection
from nomad.metainfo import SubSection, Section, Quantity
from nomad.datamodel.metainfo.workflow import Link
from .general import (
    SimulationWorkflowMethod,
    SimulationWorkflowResults,
    ParallelSimulation,
    WORKFLOW_METHOD_NAME,
    WORKFLOW_RESULTS_NAME,
)
from .thermodynamics import ThermodynamicsResults


class StrainDiagrams(ArchiveSection):
    """
    Section containing the information regarding the elastic strains.
    """

    m_def = Section(validate=False)

    type = Quantity(
        type=str,
        shape=[],
        description="""
        Kind of strain diagram. Possible values are: energy; cross-validation (cross-
        validation error); d2E (second derivative of the energy wrt the strain)
        """,
    )

    n_eta = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of strain values used in the strain diagram
        """,
    )

    n_deformations = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of deformations.
        """,
    )

    value = Quantity(
        type=np.float64,
        shape=['n_deformations', 'n_eta'],
        description="""
        Values of the energy(units:J)/d2E(units:Pa)/cross-validation (depending on the
        value of strain_diagram_type)
        """,
    )

    eta = Quantity(
        type=np.float64,
        shape=['n_deformations', 'n_eta'],
        description="""
        eta values used the strain diagrams
        """,
    )

    stress_voigt_component = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Voigt component corresponding to the strain diagram
        """,
    )

    polynomial_fit_order = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Order of the polynomial fit
        """,
    )


class ElasticMethod(SimulationWorkflowMethod):
    energy_stress_calculator = Quantity(
        type=str,
        shape=[],
        description="""
        Name of program used to calculate energy or stress.
        """,
    )

    calculation_method = Quantity(
        type=str,
        shape=[],
        description="""
        Method used to calculate elastic constants, can either be energy or stress.
        """,
    )

    elastic_constants_order = Quantity(
        type=int,
        shape=[],
        description="""
        Order of the calculated elastic constants.
        """,
    )

    fitting_error_maximum = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Maximum error in polynomial fit.
        """,
    )

    strain_maximum = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Maximum strain applied to crystal.
        """,
    )


class ElasticResults(ThermodynamicsResults):
    n_deformations = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of deformed structures used to calculate the elastic constants. This is
        determined by the symmetry of the crystal.
        """,
    )

    deformation_types = Quantity(
        type=np.str_,
        shape=['n_deformations', 6],
        description="""
        deformation types
        """,
    )

    n_strains = Quantity(
        type=np.int32,
        shape=[],
        description="""
        number of equally spaced strains applied to each deformed structure, which are
        generated between the maximum negative strain and the maximum positive one.
        """,
    )

    is_mechanically_stable = Quantity(
        type=bool,
        shape=[],
        description="""
        Indicates if structure is mechanically stable from the calculated values of the
        elastic constants.
        """,
    )

    elastic_constants_notation_matrix_second_order = Quantity(
        type=np.str_,
        shape=[6, 6],
        description="""
        Symmetry of the second-order elastic constant matrix in Voigt notation
        """,
    )

    elastic_constants_matrix_second_order = Quantity(
        type=np.float64,
        shape=[6, 6],
        unit='pascal',
        description="""
        2nd order elastic constant (stiffness) matrix in pascals
        """,
    )

    elastic_constants_matrix_third_order = Quantity(
        type=np.float64,
        shape=[6, 6, 6],
        unit='pascal',
        description="""
        3rd order elastic constant (stiffness) matrix in pascals
        """,
    )

    compliance_matrix_second_order = Quantity(
        type=np.float64,
        shape=[6, 6],
        unit='1 / pascal',
        description="""
        Elastic compliance matrix in 1/GPa
        """,
    )

    elastic_constants_gradient_matrix_second_order = Quantity(
        type=np.float64,
        shape=[18, 18],
        unit='newton',
        description="""
        gradient of the 2nd order elastic constant (stiffness) matrix in newton
        """,
    )

    bulk_modulus_voigt = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        Voigt bulk modulus
        """,
    )

    shear_modulus_voigt = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        Voigt shear modulus
        """,
    )

    bulk_modulus_reuss = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        Reuss bulk modulus
        """,
    )

    shear_modulus_reuss = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        Reuss shear modulus
        """,
    )

    bulk_modulus_hill = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        Hill bulk modulus
        """,
    )

    shear_modulus_hill = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        Hill shear modulus
        """,
    )

    young_modulus_voigt = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        Voigt Young modulus
        """,
    )

    poisson_ratio_voigt = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Voigt Poisson ratio
        """,
    )

    young_modulus_reuss = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        Reuss Young modulus
        """,
    )

    poisson_ratio_reuss = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Reuss Poisson ratio
        """,
    )

    young_modulus_hill = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        Hill Young modulus
        """,
    )

    poisson_ratio_hill = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Hill Poisson ratio
        """,
    )

    elastic_anisotropy = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Elastic anisotropy
        """,
    )

    pugh_ratio_hill = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Pugh ratio defined as the ratio between the shear modulus and bulk modulus
        """,
    )

    debye_temperature = Quantity(
        type=np.float64,
        shape=[],
        unit='kelvin',
        description="""
        Debye temperature
        """,
    )

    speed_sound_transverse = Quantity(
        type=np.float64,
        shape=[],
        unit='meter / second',
        description="""
        Speed of sound along the transverse direction
        """,
    )

    speed_sound_longitudinal = Quantity(
        type=np.float64,
        shape=[],
        unit='meter / second',
        description="""
        Speed of sound along the longitudinal direction
        """,
    )

    speed_sound_average = Quantity(
        type=np.float64,
        shape=[],
        unit='meter / second',
        description="""
        Average speed of sound
        """,
    )

    eigenvalues_elastic = Quantity(
        type=np.float64,
        shape=[6],
        unit='pascal',
        description="""
        Eigenvalues of the stiffness matrix
        """,
    )

    strain_diagrams = SubSection(sub_section=StrainDiagrams.m_def, repeats=True)


class Elastic(ParallelSimulation):
    method = SubSection(sub_section=ElasticMethod)

    results = SubSection(sub_section=ElasticResults)

    def _resolve_mechanical_stability(self):
        spacegroup, c = None, None
        try:
            spacegroup = self._systems[-1].symmetry[-1].space_group_number
            c = self.results.elastic_constants_matrix_second_order
        except Exception:
            return False

        if c is None or spacegroup is None:
            return False

        # see Phys. Rev B 90, 224104 (2014)
        res = False
        if spacegroup <= 2:  # Triclinic
            res = np.count_nonzero(c < 0)
        elif spacegroup <= 15:  # Monoclinic
            res = np.count_nonzero(c < 0)
        elif spacegroup <= 74:  # Orthorhombic
            res = (
                c[0][0] > 0
                and c[0][0] * c[1][1] > c[0][1] ** 2
                and c[0][0] * c[1][1] * c[2][2]
                + 2 * c[0][1] * c[0][2] * c[1][2]
                - c[0][0] * c[1][2] ** 2
                - c[1][1] * c[0][2] ** 2
                - c[2][2] * c[0][1] ** 2
                > 0
                and c[3][3] > 0
                and c[4][4] > 0
                and c[5][5] > 0
            )
        elif spacegroup <= 88:  # Tetragonal II
            res = c[0][0] > abs(c[0][1]) and 2 * c[0][2] ** 2 < c[2][2] * (
                c[0][0] + c[0][1]
            )
        elif spacegroup <= 142:  # Tetragonal I
            res = (
                c[0][0] > abs(c[0][1])
                and 2 * c[0][2] ** 2 < c[2][2] * (c[0][0] + c[0][1])
                and c[3][3] > 0
                and c[5][5] > 0
            )
        elif spacegroup <= 148:  # rhombohedral II
            res = (
                c[0][0] > abs(c[0][1])
                and c[3][3] > 0
                and c[0][2] ** 2 < (0.5 * c[2][2] * (c[0][0] + c[0][1]))
                and c[0][3] ** 2 + c[0][4] ** 2 < 0.5 * c[3][3] * (c[0][0] - c[0][1])
            )
        elif spacegroup <= 167:  # rhombohedral I
            res = (
                c[0][0] > abs(c[0][1])
                and c[3][3] > 0
                and c[0][2] ** 2 < 0.5 * c[2][2] * (c[0][0] + c[0][1])
                and c[0][3] ** 2 < 0.5 * c[3][3] * (c[0][0] - c[0][1])
            )
        elif spacegroup <= 194:  # hexagonal I
            res = (
                c[0][0] > abs(c[0][1])
                and 2 * c[0][2] ** 2 < c[2][2] * (c[0][0] + c[0][1])
                and c[3][3] > 0
                and c[5][5] > 0
            )
        else:  # cubic
            res = c[0][0] - c[0][1] > 0 and c[0][0] + 2 * c[0][1] > 0 and c[3][3] > 0

        return res

    def _get_maximum_fit_error(self):
        max_error = 0.0
        if len(self._calculations) == 0:
            return max_error

        for diagram in self.results.strain_diagrams:
            if diagram.type == 'cross-validation':
                error = np.amax(diagram.value)
                max_error = error if error > max_error else max_error

        return max_error

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if not self.method:
            self.method = ElasticMethod()
            self.inputs.append(Link(name=WORKFLOW_METHOD_NAME, section=self.method))

        if not self.results:
            self.results = ElasticResults()
            self.outputs.append(Link(name=WORKFLOW_RESULTS_NAME, section=self.results))

        if self.results.is_mechanically_stable is None:
            self.results.is_mechanically_stable = bool(
                self._resolve_mechanical_stability()
            )

        if self.method.fitting_error_maximum is None:
            self.method.fitting_error_maximum = self._get_maximum_fit_error()

        SimulationWorkflowResults.normalize(self.results, archive, logger)
