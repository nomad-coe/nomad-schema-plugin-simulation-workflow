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
from typing import List
import numpy as np

from nomad.datamodel.data import ArchiveSection
from nomad.metainfo import (
    SubSection,
    Section,
    Quantity,
    MEnum,
    Reference,
    MSection,
    HDF5Reference,
)
from nomad.datamodel.metainfo.workflow import Link
from runschema.system import System, AtomsGroup
from runschema.calculation import (
    RadiusOfGyration as RadiusOfGyrationCalculation,
    RadiusOfGyrationValues as RadiusOfGyrationValuesCalculation,
)
from nomad.atomutils import archive_to_universe
from nomad.atomutils import (
    calc_molecular_rdf,
    calc_molecular_mean_squared_displacements,
    calc_molecular_radius_of_gyration,
)
from .general import (
    SimulationWorkflowMethod,
    SimulationWorkflowResults,
    SerialSimulation,
    WORKFLOW_METHOD_NAME,
    WORKFLOW_RESULTS_NAME,
)
from .thermodynamics import ThermodynamicsResults


class ThermostatParameters(ArchiveSection):
    """
    Section containing the parameters pertaining to the thermostat for a molecular dynamics run.
    """

    m_def = Section(validate=False)

    thermostat_type = Quantity(
        type=MEnum(
            "andersen",
            "berendsen",
            "brownian",
            "langevin_goga",
            "langevin_schneider",
            "nose_hoover",
            "velocity_rescaling",
            "velocity_rescaling_langevin",
            "velocity_rescaling_woodcock",
            "langevin_leap_frog",
        ),
        shape=[],
        description="""
        The name of the thermostat used for temperature control. If skipped or an empty string is used, it
        means no thermostat was applied.

        Allowed values are:

        | Thermostat Name        | Description                               |

        | ---------------------- | ----------------------------------------- |

        | `""`                   | No thermostat               |

        | `"andersen"`           | H.C. Andersen, [J. Chem. Phys.
        **72**, 2384 (1980)](https://doi.org/10.1063/1.439486) |

        | `"berendsen"`          | H. J. C. Berendsen, J. P. M. Postma,
        W. F. van Gunsteren, A. DiNola, and J. R. Haak, [J. Chem. Phys.
        **81**, 3684 (1984)](https://doi.org/10.1063/1.448118) |

        | `"brownian"`           | Brownian Dynamics |

        | `"langevin_goga"`           | N. Goga, A. J. Rzepiela, A. H. de Vries,
        S. J. Marrink, and H. J. C. Berendsen, [J. Chem. Theory Comput. **8**, 3637 (2012)]
        (https://doi.org/10.1021/ct3000876) |

        | `"langevin_schneider"`           | T. Schneider and E. Stoll,
        [Phys. Rev. B **17**, 1302](https://doi.org/10.1103/PhysRevB.17.1302) |

        | `"nose_hoover"`        | S. Nosé, [Mol. Phys. **52**, 255 (1984)]
        (https://doi.org/10.1080/00268978400101201); W.G. Hoover, [Phys. Rev. A
        **31**, 1695 (1985) |

        | `"velocity_rescaling"` | G. Bussi, D. Donadio, and M. Parrinello,
        [J. Chem. Phys. **126**, 014101 (2007)](https://doi.org/10.1063/1.2408420) |

        | `"velocity_rescaling_langevin"` | G. Bussi and M. Parrinello,
        [Phys. Rev. E **75**, 056707 (2007)](https://doi.org/10.1103/PhysRevE.75.056707) |

        | `"velocity_rescaling_woodcock"` | L. V. Woodcock,
        [Chem. Phys. Lett. **10**, 257 (1971)](https://doi.org/10.1016/0009-2614(71)80281-6) |

        | `"langevin_leap_frog"` | J.A. Izaguirre, C.R. Sweet, and V.S. Pande
        [Pac Symp Biocomput. **15**, 240-251 (2010)](https://doi.org/10.1142/9789814295291_0026) |
        """,
    )

    reference_temperature = Quantity(
        type=np.float64,
        shape=[],
        unit="kelvin",
        description="""
        The target temperature for the simulation. Typically used when temperature_profile is "constant".
        """,
    )

    coupling_constant = Quantity(
        type=np.float64,
        shape=[],
        unit="s",
        description="""
        The time constant for temperature coupling. Need to describe what this means for the various
        thermostat options...
        """,
    )

    effective_mass = Quantity(
        type=np.float64,
        shape=[],
        unit="kilogram",
        description="""
        The effective or fictitious mass of the temperature resevoir.
        """,
    )

    temperature_profile = Quantity(
        type=MEnum("constant", "linear", "exponential"),
        shape=[],
        description="""
        Type of temperature control (i.e., annealing) procedure. Can be "constant" (no annealing), "linear", or "exponential".
        If linear, "temperature_update_delta" specifies the corresponding update parameter.
        If exponential, "temperature_update_factor" specifies the corresponding update parameter.
        """,
    )

    reference_temperature_start = Quantity(
        type=np.float64,
        shape=[],
        unit="kelvin",
        description="""
        The initial target temperature for the simulation. Typically used when temperature_profile is "linear" or "exponential".
        """,
    )

    reference_temperature_end = Quantity(
        type=np.float64,
        shape=[],
        unit="kelvin",
        description="""
        The final target temperature for the simulation.  Typically used when temperature_profile is "linear" or "exponential".
        """,
    )

    temperature_update_frequency = Quantity(
        type=int,
        shape=[],
        description="""
        Number of simulation steps between changing the target temperature.
        """,
    )

    temperature_update_delta = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Amount to be added (subtracted if negative) to the current reference_temperature
        at a frequency of temperature_update_frequency when temperature_profile is "linear".
        The reference temperature is then replaced by this new value until the next update.
        """,
    )

    temperature_update_factor = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Factor to be multiplied to the current reference_temperature at a frequency of temperature_update_frequency when temperature_profile is exponential.
        The reference temperature is then replaced by this new value until the next update.
        """,
    )

    step_start = Quantity(
        type=int,
        shape=[],
        description="""
        Trajectory step where this thermostating starts.
        """,
    )

    step_end = Quantity(
        type=int,
        shape=[],
        description="""
        Trajectory step number where this thermostating ends.
        """,
    )


class BarostatParameters(ArchiveSection):
    """
    Section containing the parameters pertaining to the barostat for a molecular dynamics run.
    """

    m_def = Section(validate=False)

    barostat_type = Quantity(
        type=MEnum(
            "berendsen",
            "martyna_tuckerman_tobias_klein",
            "nose_hoover",
            "parrinello_rahman",
            "stochastic_cell_rescaling",
        ),
        shape=[],
        description="""
        The name of the barostat used for temperature control. If skipped or an empty string is used, it
        means no barostat was applied.

        Allowed values are:

        | Barostat Name          | Description                               |

        | ---------------------- | ----------------------------------------- |

        | `""`                   | No thermostat               |

        | `"berendsen"`          | H. J. C. Berendsen, J. P. M. Postma,
        W. F. van Gunsteren, A. DiNola, and J. R. Haak, [J. Chem. Phys.
        **81**, 3684 (1984)](https://doi.org/10.1063/1.448118) |

        | `"martyna_tuckerman_tobias_klein"` | G.J. Martyna, M.E. Tuckerman, D.J. Tobias, and M.L. Klein,
        [Mol. Phys. **87**, 1117 (1996)](https://doi.org/10.1080/00268979600100761);
        M.E. Tuckerman, J. Alejandre, R. López-Rendón, A.L. Jochim, and G.J. Martyna,
        [J. Phys. A. **59**, 5629 (2006)](https://doi.org/10.1088/0305-4470/39/19/S18)|

        | `"nose_hoover"`        | S. Nosé, [Mol. Phys. **52**, 255 (1984)]
        (https://doi.org/10.1080/00268978400101201); W.G. Hoover, [Phys. Rev. A
        **31**, 1695 (1985) |

        | `"parrinello_rahman"`        | M. Parrinello and A. Rahman,
        [J. Appl. Phys. **52**, 7182 (1981)](https://doi.org/10.1063/1.328693);
        S. Nosé and M.L. Klein, [Mol. Phys. **50**, 1055 (1983) |

        | `"stochastic_cell_rescaling"` | M. Bernetti and G. Bussi,
        [J. Chem. Phys. **153**, 114107 (2020)](https://doi.org/10.1063/1.2408420) |
        """,
    )

    coupling_type = Quantity(
        type=MEnum("isotropic", "semi_isotropic", "anisotropic"),
        shape=[],
        description="""
        Describes the symmetry of pressure coupling. Specifics can be inferred from the `coupling constant`

        | Type          | Description                               |

        | ---------------------- | ----------------------------------------- |

        | `isotropic`          | Identical coupling in all directions. |

        | `semi_isotropic` | Identical coupling in 2 directions. |

        | `anisotropic`        | General case. |
        """,
    )

    reference_pressure = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit="pascal",
        description="""
        The target pressure for the simulation, stored in a 3x3 matrix, indicating the values for individual directions
        along the diagonal, and coupling between directions on the off-diagonal. Typically used when pressure_profile is "constant".
        """,
    )

    coupling_constant = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit="s",
        description="""
        The time constants for pressure coupling, stored in a 3x3 matrix, indicating the values for individual directions
        along the diagonal, and coupling between directions on the off-diagonal. 0 values along the off-diagonal
        indicate no-coupling between these directions.
        """,
    )

    compressibility = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit="1 / pascal",
        description="""
        An estimate of the system's compressibility, used for box rescaling, stored in a 3x3 matrix indicating the values for individual directions
        along the diagonal, and coupling between directions on the off-diagonal. If None, it may indicate that these values
        are incorporated into the coupling_constant, or simply that the software used uses a fixed value that is not available in
        the input/output files.
        """,
    )

    pressure_profile = Quantity(
        type=MEnum("constant", "linear", "exponential"),
        shape=[],
        description="""
        Type of pressure control procedure. Can be "constant" (no annealing), "linear", or "exponential".
        If linear, "pressure_update_delta" specifies the corresponding update parameter.
        If exponential, "pressure_update_factor" specifies the corresponding update parameter.
        """,
    )

    reference_pressure_start = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit="pascal",
        description="""
        The initial target pressure for the simulation, stored in a 3x3 matrix, indicating the values for individual directions
        along the diagonal, and coupling between directions on the off-diagonal. Typically used when pressure_profile is "linear" or "exponential".
        """,
    )

    reference_pressure_end = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit="pascal",
        description="""
        The final target pressure for the simulation, stored in a 3x3 matrix, indicating the values for individual directions
        along the diagonal, and coupling between directions on the off-diagonal.  Typically used when pressure_profile is "linear" or "exponential".
        """,
    )

    pressure_update_frequency = Quantity(
        type=int,
        shape=[],
        description="""
        Number of simulation steps between changing the target pressure.
        """,
    )

    pressure_update_delta = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Amount to be added (subtracted if negative) to the current reference_pressure
        at a frequency of pressure_update_frequency when pressure_profile is "linear".
        The pressure temperature is then replaced by this new value until the next update.
        """,
    )

    pressure_update_factor = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Factor to be multiplied to the current reference_pressure at a frequency of pressure_update_frequency when pressure_profile is exponential.
        The reference pressure is then replaced by this new value until the next update.
        """,
    )

    step_start = Quantity(
        type=int,
        shape=[],
        description="""
        Trajectory step where this barostating starts.
        """,
    )

    step_end = Quantity(
        type=int,
        shape=[],
        description="""
        Trajectory step number where this barostating ends.
        """,
    )


class Lambdas(MSection):
    """
    Section for storing all lambda parameters for free energy perturbation
    """

    m_def = Section(validate=False)

    kind = Quantity(
        type=MEnum(
            "output", "coulomb", "vdw", "bonded", "restraint", "mass", "temperature"
        ),
        shape=[],
        description="""
        The type of lambda interpolation

        Allowed values are:

        | kind          | Description                               |

        | ---------------------- | ----------------------------------------- |

        | `"output"`           | Lambdas for the free energy outputs saved.
                                    These will also act as a default in case some
                                    relevant lambdas are not specified. |

        | `"coulomb"`          | Lambdas for interpolating electrostatic interactions. |

        | `"vdw"`              | Lambdas for interpolating van der Waals interactions. |

        | `"bonded"`           | Lambdas for interpolating all intramolecular interactions. |

        | `"restraint"`        | Lambdas for interpolating restraints. |

        | `"mass"`             | Lambdas for interpolating masses. |

        | `"temperature"`      | Lambdas for interpolating temperature. |
        """,
    )

    value = Quantity(
        type=np.float64,
        shape=[],
        description="""
        List of lambdas.
        """,
    )


class FreeEnergyCalculationParameters(ArchiveSection):
    """
    Section containing the parameters pertaining to a free energy calculation workflow
    that interpolates between two system states (defined via the interpolation parameter lambda).
    The parameters are stored for each molecular dynamics run separately, to be referenced
    by the overarching workflow.
    """

    m_def = Section(validate=False)

    type = Quantity(
        type=MEnum("alchemical", "umbrella_sampling"),
        shape=[],
        description="""
        Specifies the type of workflow.
        """,
    )

    lambdas = SubSection(
        sub_section=Lambdas.m_def,
        description="""
        Contains the lists of lambda values defined for the interpolation of the system.
        """,
        repeats=True,
    )

    lambda_index = Quantity(
        type=int,
        shape=[],
        description="""
        The index of the lambda in `lambdas` corresponding to the state of the current simulation.
        """,
    )

    atom_indices = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description="""
        List of atom indices involved in the interpolation.
        """,
    )

    initial_state_vdw = Quantity(
        type=MEnum("on", "off"),
        shape=[],
        description="""
        Specifies whether vdw interactions are on or off in the initial state (i.e., lambda = 0).
        """,
    )

    final_state_vdw = Quantity(
        type=MEnum("on", "off"),
        shape=[],
        description="""
        Specifies whether vdw interactions are on or off in the final state (i.e., lambda = 0).
        """,
    )

    initial_state_coloumb = Quantity(
        type=MEnum("on", "off"),
        shape=[],
        description="""
        Specifies whether vdw interactions are on or off in the initial state (i.e., lambda = 0).
        """,
    )

    final_state_coloumb = Quantity(
        type=MEnum("on", "off"),
        shape=[],
        description="""
        Specifies whether vdw interactions are on or off in the final state (i.e., lambda = 0).
        """,
    )

    initial_state_bonded = Quantity(
        type=MEnum("on", "off"),
        shape=[],
        description="""
        Specifies whether bonded interactions are on or off in the initial state (i.e., lambda = 0).
        """,
    )

    final_state_bonded = Quantity(
        type=MEnum("on", "off"),
        shape=[],
        description="""
        Specifies whether bonded interactions are on or off in the final state (i.e., lambda = 0).
        """,
    )


class MolecularDynamicsMethod(SimulationWorkflowMethod):
    thermodynamic_ensemble = Quantity(
        type=MEnum("NVE", "NVT", "NPT", "NPH"),
        shape=[],
        description="""
        The type of thermodynamic ensemble that was simulated.

        Allowed values are:

        | Thermodynamic Ensemble          | Description                               |

        | ---------------------- | ----------------------------------------- |

        | `"NVE"`           | Constant number of particles, volume, and energy |

        | `"NVT"`           | Constant number of particles, volume, and temperature |

        | `"NPT"`           | Constant number of particles, pressure, and temperature |

        | `"NPH"`           | Constant number of particles, pressure, and enthalpy |
        """,
    )

    integrator_type = Quantity(
        type=MEnum(
            "brownian",
            "conjugant_gradient",
            "langevin_goga",
            "langevin_schneider",
            "leap_frog",
            "rRESPA_multitimescale",
            "velocity_verlet",
            "langevin_leap_frog",
        ),
        shape=[],
        description="""
        Name of the integrator.

        Allowed values are:

        | Integrator Name          | Description                               |

        | ---------------------- | ----------------------------------------- |

        | `"langevin_goga"`           | N. Goga, A. J. Rzepiela, A. H. de Vries,
        S. J. Marrink, and H. J. C. Berendsen, [J. Chem. Theory Comput. **8**, 3637 (2012)]
        (https://doi.org/10.1021/ct3000876) |

        | `"langevin_schneider"`           | T. Schneider and E. Stoll,
        [Phys. Rev. B **17**, 1302](https://doi.org/10.1103/PhysRevB.17.1302) |

        | `"leap_frog"`          | R.W. Hockney, S.P. Goel, and J. Eastwood,
        [J. Comp. Phys. **14**, 148 (1974)](https://doi.org/10.1016/0021-9991(74)90010-2) |

        | `"velocity_verlet"` | W.C. Swope, H.C. Andersen, P.H. Berens, and K.R. Wilson,
        [J. Chem. Phys. **76**, 637 (1982)](https://doi.org/10.1063/1.442716) |

        | `"rRESPA_multitimescale"` | M. Tuckerman, B. J. Berne, and G. J. Martyna
        [J. Chem. Phys. **97**, 1990 (1992)](https://doi.org/10.1063/1.463137) |

        | `"langevin_leap_frog"` | J.A. Izaguirre, C.R. Sweet, and V.S. Pande
        [Pac Symp Biocomput. **15**, 240-251 (2010)](https://doi.org/10.1142/9789814295291_0026) |
        """,
    )

    integration_timestep = Quantity(
        type=np.float64,
        shape=[],
        unit="s",
        description="""
        The timestep at which the numerical integration is performed.
        """,
    )

    n_steps = Quantity(
        type=int,
        shape=[],
        description="""
        Number of timesteps performed.
        """,
    )

    coordinate_save_frequency = Quantity(
        type=int,
        shape=[],
        description="""
        The number of timesteps between saving the coordinates.
        """,
    )

    velocity_save_frequency = Quantity(
        type=int,
        shape=[],
        description="""
        The number of timesteps between saving the velocities.
        """,
    )

    force_save_frequency = Quantity(
        type=int,
        shape=[],
        description="""
        The number of timesteps between saving the forces.
        """,
    )

    thermodynamics_save_frequency = Quantity(
        type=int,
        shape=[],
        description="""
        The number of timesteps between saving the thermodynamic quantities.
        """,
    )

    thermostat_parameters = SubSection(
        sub_section=ThermostatParameters.m_def, repeats=True
    )

    barostat_parameters = SubSection(sub_section=BarostatParameters.m_def, repeats=True)

    free_energy_calculation_parameters = SubSection(
        sub_section=FreeEnergyCalculationParameters.m_def, repeats=True
    )


class Property(ArchiveSection):
    """
    Generic parent section for all property types.
    """

    m_def = Section(validate=False)

    type = Quantity(
        type=MEnum("molecular", "atomic"),
        shape=[],
        description="""
        Describes if the observable is calculated at the molecular or atomic level.
        """,
    )

    label = Quantity(
        type=str,
        shape=[],
        description="""
        Name or description of the property.
        """,
    )

    error_type = Quantity(
        type=str,
        shape=[],
        description="""
        Describes the type of error reported for this observable.
        """,
    )


class PropertyValues(MSection):
    """
    Generic parent section for information regarding the values of a property.
    """

    m_def = Section(validate=False)

    label = Quantity(
        type=str,
        shape=[],
        description="""
        Describes the atoms or molecule types involved in determining the property.
        """,
    )

    errors = Quantity(
        type=np.float64,
        shape=["*"],
        description="""
        Error associated with the determination of the property.
        """,
    )


class EnsemblePropertyValues(PropertyValues):
    """
    Generic section containing information regarding the values of an ensemble property.
    """

    m_def = Section(validate=False)

    n_bins = Quantity(
        type=int,
        shape=[],
        description="""
        Number of bins.
        """,
    )

    frame_start = Quantity(
        type=int,
        shape=[],
        description="""
        Trajectory frame number where the ensemble averaging starts.
        """,
    )

    frame_end = Quantity(
        type=int,
        shape=[],
        description="""
        Trajectory frame number where the ensemble averaging ends.
        """,
    )

    bins_magnitude = Quantity(
        type=np.float64,
        shape=["n_bins"],
        description="""
        Values of the variable along which the property is calculated.
        """,
    )

    bins_unit = Quantity(
        type=str,
        shape=[],
        description="""
        Unit of the given bins, using UnitRegistry() notation.
        """,
    )

    value_magnitude = Quantity(
        type=np.float64,
        shape=["n_bins"],
        description="""
        Values of the property.
        """,
    )

    value_unit = Quantity(
        type=str,
        shape=[],
        description="""
        Unit of the property, using UnitRegistry() notation.
        """,
    )


class RadialDistributionFunctionValues(EnsemblePropertyValues):
    """
    Section containing information regarding the values of
    radial distribution functions (rdfs).
    """

    m_def = Section(validate=False)

    bins = Quantity(
        type=np.float64,
        shape=["n_bins"],
        unit="m",
        description="""
        Distances along which the rdf was calculated.
        """,
    )

    value = Quantity(
        type=np.float64,
        shape=["n_bins"],
        description="""
        Values of the property.
        """,
    )


class EnsembleProperty(Property):
    """
    Generic section containing information about a calculation of any static observable
    from a trajectory (i.e., from an ensemble average).
    """

    m_def = Section(validate=False)

    n_smooth = Quantity(
        type=int,
        shape=[],
        description="""
        Number of bins over which the running average was computed for
        the observable `values'.
        """,
    )

    n_variables = Quantity(
        type=int,
        shape=[],
        description="""
        Number of variables along which the property is determined.
        """,
    )

    variables_name = Quantity(
        type=str,
        shape=["n_variables"],
        description="""
        Name/description of the independent variables along which the observable is defined.
        """,
    )

    ensemble_property_values = SubSection(
        sub_section=EnsemblePropertyValues.m_def, repeats=True
    )


class RadialDistributionFunction(EnsembleProperty):
    """
    Section containing information about the calculation of
    radial distribution functions (rdfs).
    """

    m_def = Section(validate=False)

    _rdf_results = None

    radial_distribution_function_values = SubSection(
        sub_section=RadialDistributionFunctionValues.m_def, repeats=True
    )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if self._rdf_results:
            self.type = self._rdf_results.get("type")
            self.n_smooth = self._rdf_results.get("n_smooth")
            self.n_prune = self._rdf_results.get("n_prune")
            self.n_variables = 1
            self.variables_name = ["distance"]
            for i_pair, pair_type in enumerate(self._rdf_results.get("types", [])):
                sec_rdf_values = self.m_create(RadialDistributionFunctionValues)
                sec_rdf_values.label = str(pair_type)
                sec_rdf_values.n_bins = len(
                    self._rdf_results.get("bins", [[]] * i_pair)[i_pair]
                )
                sec_rdf_values.bins = self._rdf_results.get("bins", [[]] * i_pair)[
                    i_pair
                ]
                sec_rdf_values.value = self._rdf_results.get("value", [[]] * i_pair)[
                    i_pair
                ]
                sec_rdf_values.frame_start = self._rdf_results.get(
                    "frame_start", [[]] * i_pair
                )[i_pair]
                sec_rdf_values.frame_end = self._rdf_results.get(
                    "frame_end", [[]] * i_pair
                )[i_pair]


class TrajectoryProperty(Property):
    """
    Generic section containing information about a calculation of any observable
    defined and stored at each individual frame of a trajectory.
    """

    m_def = Section(validate=False)

    n_frames = Quantity(
        type=int,
        shape=[],
        description="""
        Number of frames for which the observable is stored.
        """,
    )

    frames = Quantity(
        type=np.int32,
        shape=["n_frames"],
        description="""
        Frames for which the observable is stored.
        """,
    )

    times = Quantity(
        type=np.float64,
        shape=["n_frames"],
        unit="s",
        description="""
        Times for which the observable is stored.
        """,
    )

    value_magnitude = Quantity(
        type=np.float64,
        shape=["n_times"],
        description="""
        Values of the property.
        """,
    )

    value_unit = Quantity(
        type=str,
        shape=[],
        description="""
        Unit of the property, using UnitRegistry() notation.
        """,
    )

    errors = Quantity(
        type=np.float64,
        shape=["*"],
        description="""
        Error associated with the determination of the property.
        """,
    )


# TODO Rg + TrajectoryPropery should be removed from workflow. All properties dependent on a single configuration should be store in calculation
class RadiusOfGyration(TrajectoryProperty):
    """
    Section containing information about the calculation of
    radius of gyration (Rg).
    """

    m_def = Section(validate=False)

    _rg_results = None

    atomsgroup_ref = Quantity(
        type=Reference(AtomsGroup.m_def),
        shape=[1],
        description="""
        References to the atoms_group section containing the molecule for which Rg was calculated.
        """,
    )

    value = Quantity(
        type=np.float64,
        shape=["n_frames"],
        unit="m",
        description="""
        Values of the property.
        """,
    )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if self._rg_results:
            self.type = self._rg_results.get("type")
            self.label = self._rg_results.get("label")
            self.atomsgroup_ref = self._rg_results.get("atomsgroup_ref")
            self.n_frames = self._rg_results.get("n_frames")
            self.times = self._rg_results.get("times")
            self.value = self._rg_results.get("value")


class FreeEnergyCalculations(TrajectoryProperty):
    """
    Section containing information regarding the instantaneous (i.e., for a single configuration)
    values of free energies calculated via thermodynamic perturbation.
    The values stored are actually infinitesimal changes in the free energy, determined as derivatives
    of the Hamiltonian with respect to the coupling parameter (lambda) defining each state for the perturbation.
    """

    m_def = Section(validate=False)

    method_ref = Quantity(
        type=Reference(FreeEnergyCalculationParameters.m_def),
        shape=[],
        description="""
        Links the free energy results with the method parameters.
        """,
    )

    lambda_index = Quantity(
        type=int,
        shape=[],
        description="""
        Index of the lambda state for the present simulation within the free energy calculation workflow.
        I.e., lambda = method_ref.lambdas.values[lambda_index]
        """,
    )

    n_states = Quantity(
        type=int,
        shape=[],
        description="""
        Number of states defined for the interpolation of the system, as indicate in `method_ref`.
        """,
    )

    value_unit = Quantity(
        type=str,
        shape=[],
        description="""
        Unit of the property, using UnitRegistry() notation.
        In this case, the unit corresponds to all `value` properties stored within this section.
        """,
    )

    value_total_energy_magnitude = Quantity(
        type=HDF5Reference,
        shape=[],
        description="""
        Value of the total energy for the present lambda state. The expected dimensions are ["n_frames"].
        This quantity is a reference to the data (file+path), which is stored in an HDF5 file for efficiency.
        """,
    )

    value_PV_energy_magnitude = Quantity(
        type=HDF5Reference,
        shape=[],
        description="""
        Value of the pressure-volume energy (i.e., P*V) for the present lambda state. The expected dimensions are ["n_frames"].
        This quantity is a reference to the data (file+path), which is stored in an HDF5 file for efficiency.
        """,
    )

    value_total_energy_differences_magnitude = Quantity(
        type=HDF5Reference,
        shape=[],
        description="""
        Values correspond to the difference in total energy between each specified lambda state
        and the reference state, which corresponds to the value of lambda of the current simulation.
        The expected dimensions are ["n_frames", "n_states"].
        This quantity is a reference to the data (file+path), which is stored in an HDF5 file for efficiency.
        """,
    )

    value_total_energy_derivative_magnitude = Quantity(
        type=HDF5Reference,
        shape=[],
        description="""
        Value of the derivative of the total energy with respect to lambda, evaluated for the current
        lambda state. The expected dimensions are ["n_frames"].
        This quantity is a reference to the data (file+path), which is stored in an HDF5 file for efficiency.
        """,
    )


class DiffusionConstantValues(PropertyValues):
    """
    Section containing information regarding the diffusion constants.
    """

    m_def = Section(validate=False)

    value = Quantity(
        type=np.float64,
        shape=[],
        unit="m^2/s",
        description="""
        Values of the diffusion constants.
        """,
    )

    error_type = Quantity(
        type=str,
        shape=[],
        description="""
        Describes the type of error reported for this observable.
        """,
    )


class CorrelationFunctionValues(PropertyValues):
    """
    Generic section containing information regarding the values of a correlation function.
    """

    m_def = Section(validate=False)

    n_times = Quantity(
        type=int,
        shape=[],
        description="""
        Number of times windows for the calculation of the correlation function.
        """,
    )

    times = Quantity(
        type=np.float64,
        shape=["n_times"],
        unit="s",
        description="""
        Time windows used for the calculation of the correlation function.
        """,
    )

    value_magnitude = Quantity(
        type=np.float64,
        shape=["n_times"],
        description="""
        Values of the property.
        """,
    )

    value_unit = Quantity(
        type=str,
        shape=[],
        description="""
        Unit of the property, using UnitRegistry() notation.
        """,
    )


class MeanSquaredDisplacementValues(CorrelationFunctionValues):
    """
    Section containing information regarding the values of a mean squared displacements (msds).
    """

    m_def = Section(validate=False)

    times = Quantity(
        type=np.float64,
        shape=["n_times"],
        unit="s",
        description="""
        Time windows used for the calculation of the msds.
        """,
    )

    value = Quantity(
        type=np.float64,
        shape=["n_times"],
        unit="m^2",
        description="""
        Mean squared displacement values.
        """,
    )

    errors = Quantity(
        type=np.float64,
        shape=["*"],
        description="""
        Error associated with the determination of the msds.
        """,
    )

    diffusion_constant = SubSection(
        sub_section=DiffusionConstantValues.m_def, repeats=False
    )


class CorrelationFunction(Property):
    """
    Generic section containing information about a calculation of any time correlation
    function from a trajectory.
    """

    m_def = Section(validate=False)

    direction = Quantity(
        type=MEnum("x", "y", "z", "xy", "yz", "xz", "xyz"),
        shape=[],
        description="""
        Describes the direction in which the correlation function was calculated.
        """,
    )

    correlation_function_values = SubSection(
        sub_section=CorrelationFunctionValues.m_def, repeats=True
    )


class MeanSquaredDisplacement(CorrelationFunction):
    """
    Section containing information about a calculation of any mean squared displacements (msds).
    """

    m_def = Section(validate=False)

    _msd_results = None

    mean_squared_displacement_values = SubSection(
        sub_section=MeanSquaredDisplacementValues.m_def, repeats=True
    )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if not self._msd_results:
            return

        self.type = self._msd_results.get("type")
        self.direction = self._msd_results.get("direction")
        for i_type, moltype in enumerate(self._msd_results.get("types", [])):
            sec_msd_values = self.m_create(MeanSquaredDisplacementValues)
            sec_msd_values.label = str(moltype)
            sec_msd_values.n_times = len(
                self._msd_results.get("times", [[]] * i_type)[i_type]
            )
            sec_msd_values.times = (
                self._msd_results["times"][i_type]
                if self._msd_results.get("times") is not None
                else []
            )
            sec_msd_values.value = (
                self._msd_results["value"][i_type]
                if self._msd_results.get("value") is not None
                else []
            )
            sec_diffusion = sec_msd_values.m_create(DiffusionConstantValues)
            sec_diffusion.value = (
                self._msd_results["diffusion_constant"][i_type]
                if self._msd_results.get("diffusion_constant") is not None
                else []
            )
            sec_diffusion.error_type = "Pearson correlation coefficient"
            sec_diffusion.errors = (
                self._msd_results["error_diffusion_constant"][i_type]
                if self._msd_results.get("error_diffusion_constant") is not None
                else []
            )


class MolecularDynamicsResults(ThermodynamicsResults):
    finished_normally = Quantity(
        type=bool,
        shape=[],
        description="""
        Indicates if calculation terminated normally.
        """,
    )

    n_steps = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of trajectory steps""",
    )

    trajectory = Quantity(
        type=Reference(System),
        shape=["n_steps"],
        description="""
        Reference to the system of each step in the trajectory.
        """,
    )

    radial_distribution_functions = SubSection(
        sub_section=RadialDistributionFunction, repeats=True
    )

    ensemble_properties = SubSection(sub_section=EnsembleProperty, repeats=True)

    correlation_functions = SubSection(sub_section=CorrelationFunction, repeats=True)

    radial_distribution_functions = SubSection(
        sub_section=RadialDistributionFunction, repeats=True
    )

    radius_of_gyration = SubSection(sub_section=RadiusOfGyration, repeats=True)

    mean_squared_displacements = SubSection(
        sub_section=MeanSquaredDisplacement, repeats=True
    )

    free_energy_calculations = SubSection(
        sub_section=FreeEnergyCalculations, repeats=True
    )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        universe = archive_to_universe(archive)
        if universe is None:
            return

        # calculate molecular radial distribution functions
        if not self.radial_distribution_functions:
            n_traj_split = (
                10  # number of intervals to split trajectory into for averaging
            )
            interval_indices = []  # 2D array specifying the groups of the n_traj_split intervals to be averaged
            # first 20% of trajectory
            interval_indices.append(np.arange(int(n_traj_split * 0.20)))
            # last 80% of trajectory
            interval_indices.append(np.arange(n_traj_split)[len(interval_indices[0]) :])
            # last 60% of trajectory
            interval_indices.append(
                np.arange(n_traj_split)[len(interval_indices[0]) * 2 :]
            )
            # last 40% of trajectory
            interval_indices.append(
                np.arange(n_traj_split)[len(interval_indices[0]) * 3 :]
            )

            n_prune = int(universe.trajectory.n_frames / len(archive.run[-1].system))
            rdf_results = calc_molecular_rdf(
                universe,
                n_traj_split=n_traj_split,
                n_prune=n_prune,
                interval_indices=interval_indices,
            )
            if rdf_results:
                sec_rdfs = RadialDistributionFunction()
                sec_rdfs._rdf_results = rdf_results
                sec_rdfs.normalize(archive, logger)
                self.radial_distribution_functions.append(sec_rdfs)

        # calculate the molecular mean squared displacements
        if not self.mean_squared_displacements:
            msd_results = calc_molecular_mean_squared_displacements(universe)
            if msd_results:
                sec_msds = MeanSquaredDisplacement()
                sec_msds._msd_results = msd_results
                sec_msds.normalize(archive, logger)
                self.mean_squared_displacements.append(sec_msds)

        # calculate radius of gyration for polymers
        try:
            sec_systems = archive.run[-1].system
            sec_system = sec_systems[0]
            sec_calc = archive.run[-1].calculation
            sec_calc = sec_calc if sec_calc is not None else []
        except Exception:
            return

        flag_rgs = False
        for calc in sec_calc:
            if calc.get("radius_of_gyration"):
                flag_rgs = True
                break  # TODO Should transfer Rg's to workflow results if they are already supplied in calculation

        if not flag_rgs:
            sec_rgs_calc = None
            system_topology = sec_system.atoms_group
            rg_results = calc_molecular_radius_of_gyration(universe, system_topology)
            for rg in rg_results:
                n_frames = rg.get("n_frames")
                if len(sec_systems) != n_frames:
                    self.logger.warning(
                        "Mismatch in length of system references in calculation and calculated Rg values."
                        "Will not store Rg values under calculation section"
                    )
                    continue

                sec_rgs = RadiusOfGyration()
                sec_rgs._rg_results = rg
                sec_rgs.normalize(archive, logger)
                self.radius_of_gyration.append(sec_rgs)

                for calc in sec_calc:
                    if not calc.system_ref:
                        continue
                    sys_ind = calc.system_ref.m_parent_index
                    sec_rgs_calc = calc.radius_of_gyration
                    if not sec_rgs_calc:
                        sec_rgs_calc = calc.m_create(RadiusOfGyrationCalculation)
                        sec_rgs_calc.kind = rg.get("type")
                    else:
                        sec_rgs_calc = sec_rgs_calc[0]
                    sec_rg_values = sec_rgs_calc.m_create(
                        RadiusOfGyrationValuesCalculation
                    )
                    sec_rg_values.atomsgroup_ref = rg.get("atomsgroup_ref")
                    sec_rg_values.label = rg.get("label")
                    sec_rg_values.value = rg.get("value")[sys_ind]


class MolecularDynamics(SerialSimulation):
    method = SubSection(sub_section=MolecularDynamicsMethod)

    results = SubSection(sub_section=MolecularDynamicsResults)

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if not self.method:
            self.method = MolecularDynamicsMethod()
            self.inputs.append(Link(name=WORKFLOW_METHOD_NAME, section=self.method))

        if not self.results:
            self.results = MolecularDynamicsResults()
            self.outputs.append(Link(name=WORKFLOW_RESULTS_NAME, section=self.results))

        if self.results.trajectory is None and self._systems:
            self.results.trajectory = self._systems

        SimulationWorkflowResults.normalize(self.results, archive, logger)
