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
from runschema.method import BSE as BSEMethodology
from runschema.calculation import Spectra
from .general import (
    SimulationWorkflowResults,
    SimulationWorkflowMethod,
    ParallelSimulation,
)


class PhotonPolarizationResults(SimulationWorkflowResults):
    """Groups all polarization outputs: spectrum."""

    n_polarizations = Quantity(
        type=np.int32,
        description="""
        Number of polarizations for the phonons used for the calculations.
        """,
    )

    spectrum_polarization = Quantity(
        type=Reference(Spectra),
        shape=['n_polarizations'],
        description="""
        Spectrum for a given polarization of the photon.
        """,
    )


class PhotonPolarizationMethod(SimulationWorkflowMethod):
    """Defines the full macroscopic dielectric tensor methodology: BSE method reference."""

    # TODO add TDDFT methodology reference.

    bse_method_ref = Quantity(
        type=Reference(BSEMethodology),
        description="""
        BSE methodology reference.
        """,
    )


class PhotonPolarization(ParallelSimulation):
    """The PhotonPolarization workflow is generated in an extra EntryArchive FOR all polarization
    EntryArchives present in the upload. It groups them for a set of given method parameters.

    This entry is also recognized as the full macroscopic dielectric tensor entry (e.g. calculated
    via BSE).
    """

    method = SubSection(sub_section=PhotonPolarizationMethod)

    results = SubSection(sub_section=PhotonPolarizationResults)

    def normalize(self, archive, logger):
        super().normalize(archive, logger)
