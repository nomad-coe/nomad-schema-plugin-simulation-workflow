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
from ase.eos import EquationOfState as aseEOS

from nomad.atomutils import get_volume
from nomad.datamodel.data import ArchiveSection
from nomad.units import ureg
from nomad.metainfo import SubSection, Section, Quantity, MProxy
from nomad.datamodel.metainfo.workflow import Link
from .general import (
    SimulationWorkflowMethod,
    SimulationWorkflowResults,
    ParallelSimulation,
    WORKFLOW_METHOD_NAME,
    WORKFLOW_RESULTS_NAME,
)
from .single_point import SinglePoint
from runschema.run import Run, Program
from runschema.system import System


class EquationOfStateMethod(SimulationWorkflowMethod):
    energy_calculator = Quantity(
        type=str,
        shape=[],
        description="""
        Name of program used to calculate energy.
        """,
    )


class EOSFit(ArchiveSection):
    """
    Section containing results of an equation of state fit.
    """

    m_def = Section(validate=False)

    function_name = Quantity(
        type=str,
        shape=[],
        description="""
        Specifies the function used to perform the fitting of the volume-energy data. Value
        can be one of birch_euler, birch_lagrange, birch_murnaghan, mie_gruneisen,
        murnaghan, pack_evans_james, poirier_tarantola, tait, vinet.
        """,
    )

    fitted_energies = Quantity(
        type=np.float64,
        shape=['n_points'],
        unit='joule',
        description="""
        Array of the fitted energies corresponding to each volume.
        """,
    )

    bulk_modulus = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        Calculated value of the bulk modulus by fitting the volume-energy data.
        """,
    )

    bulk_modulus_derivative = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Calculated value of the pressure derivative of the bulk modulus.
        """,
    )

    equilibrium_volume = Quantity(
        type=np.float64,
        shape=[],
        unit='m ** 3',
        description="""
        Calculated value of the equilibrium volume by fitting the volume-energy data.
        """,
    )

    equilibrium_energy = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        Calculated value of the equilibrium energy by fitting the volume-energy data.
        """,
    )

    rms_error = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Root-mean squared value of the error in the fitting.
        """,
    )


class EquationOfStateResults(SimulationWorkflowResults):
    n_points = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of volume-energy pairs in data.
        """,
    )

    volumes = Quantity(
        type=np.float64,
        shape=['n_points'],
        unit='m ** 3',
        description="""
        Array of volumes per atom for which the energies are evaluated.
        """,
    )

    energies = Quantity(
        type=np.float64,
        shape=['n_points'],
        unit='joule',
        description="""
        Array of energies corresponding to each volume.
        """,
    )

    eos_fit = SubSection(sub_section=EOSFit.m_def, repeats=True)


class EquationOfState(ParallelSimulation):
    method = SubSection(sub_section=EquationOfStateMethod)

    results = SubSection(sub_section=EquationOfStateResults)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_archive_paths = {
            'input': 'run/0/system/-1',
            'task': 'workflow2',
        }

    # def get_default_archive_path(self, raw_proxy_value, section_type='') -> str:
    #     """
    #     Returns a certain archive path if the raw proxy value points to the root of the archive.
    #     """
    #     if raw_proxy_value is None:
    #         return ''

    #     if '#/' in raw_proxy_value:
    #         _, after = raw_proxy_value.split('#/', 1)
    #         if after:
    #             return ''
    #         else:
    #             return self.default_archive_paths.get(section_type, '')
    #     else:
    #         return ''

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        logger.warning(f'self.tasks: {self.tasks}')
        logger.warning(f'self.inputs: {self.inputs}')

        flag_input_structure = False
        input_path_global = ''
        archive_root = None
        # find the input structure
        if self.inputs:
            for input_item in self.inputs:
                input_section = input_item.section.m_resolved()
                # TODO - I need an alternative method to get the full input section path
                print(f'is MProxy: {isinstance(input_section, MProxy)}')
                # ! m_proxy_value is not available for "noraml sections"
                archive_root = archive.m_root()
                archive_metadata = archive_root.metadata if archive_root else None
                input_path_global = ''
                if isinstance(input_section, MProxy):
                    input_path_global = input_section.m_proxy_value
                elif archive_metadata:
                    upload_id = archive_metadata.upload_id
                    entry_id = archive_metadata.entry_id
                    input_path = input_section.m_path()
                    input_path_global = (
                        f'../{upload_id}/archive/{entry_id}#/{input_path}'
                        if upload_id and entry_id and input_path
                        else ''
                    )

                # default_path = self.get_default_archive_path(
                #     input_path_global, section_type='input'
                # )
                # logger.warning(f'default_path: {default_path}')
                # if default_path != '':
                #     archive_root = archive.m_context.resolve_archive(input_path_global)
                #     input_section = archive_root.m_resolve(default_path)
                if not isinstance(input_section, System):
                    continue

                flag_input_structure = True
                # input_proxy_value = input_path_global + default_path
                system_index = input_section.m_parent_index
                run_section = input_section.m_parent
                run_index = run_section.m_parent_index
                input_name = input_item.name
                if archive_root:
                    if system_index == -1:
                        system_index = len(archive_root.run[run_index].system) - 1
                if not archive.run:
                    run = Run(program=Program())
                    try:
                        run.system.extend([input_section])
                        run.method.extend(archive_root.run[run_index].method)
                        for calc in archive_root.run[run_index].calculation:
                            if calc.system_ref.m_parent_index == system_index:
                                run.calculation.extend([calc])
                                break
                    except Exception:
                        logger.warning(
                            'Failed to create run section from input structure. '
                        )

                    archive.run.append(run)

                break

        if not flag_input_structure:
            logger.warning('No input structure found in EOS workflow normalizer.')

        if not self.method:
            self.method = EquationOfStateMethod()
            self.inputs.append(Link(name=WORKFLOW_METHOD_NAME, section=self.method))

        if not self.results:
            self.results = EquationOfStateResults()
            self.outputs.append(Link(name=WORKFLOW_RESULTS_NAME, section=self.results))

        #! Causing test to fail
        try:
            task_archives = [task.task.m_root() for task in self.tasks]
            assert all(
                isinstance(task_archive.workflow2, SinglePoint)
                for task_archive in task_archives
            )
        except Exception:
            logger.warning(
                'Not all tasks are SinglePoints or failed to retrieve task archives. EOS workflow may be incomplete or incorrect.'
            )
            return

        for task in self.tasks:
            # TODO - I need an alternative method to get the full input section path
            # ! m_proxy_value is not available for "noraml sections"
            # logger.warning(f'task: {task.task}')
            # logger.warning(f'task.section: {task.task.section}')
            # raw_proxy_value = task.section.m_proxy_value
            # default_path = self.get_default_archive_path(
            #     raw_proxy_value, section_type='task'
            # )
            # if default_path:
            #     # replace the task section with the default for tasks
            #     # task.section = task.section.m_xpath(default_path)
            #     archive_root = archive.m_context.resolve_archive(raw_proxy_value)
            #     task.section = archive_root.m_resolve(default_path)

            # TODO - Add global output to each task output?

            # TODO - I need an alternative method to get the full input section path
            # ! m_proxy_value is not available for "noraml sections"
            if input_path_global:
                input_proxy_values = [
                    input.section.m_proxy_value for input in task.inputs
                ]
                if input_path_global in input_proxy_values:
                    index = input_proxy_values.index(input_path_global)
                    task.inputs[index].name = input_name
                else:
                    task.inputs.append(Link(name=input_name, section=input_section))

        if not self._calculations:
            # try to get calculations from tasks (in case of instantiation from workflow yaml)
            try:
                self._calculations = [
                    task.task.results.calculations_ref[0] for task in self.tasks
                ]
            except Exception:
                pass

        if not self._systems:
            # try to get systems from calculations (in case of instantiation from workflow yaml)
            try:
                self._systems = [calc.system_ref for calc in self._calculations]
            except Exception:
                pass

        if self.results.energies is None:
            try:
                self.results.energies = [
                    calc.energy.total.value.magnitude for calc in self._calculations
                ]
            except Exception:
                pass

        if self.results.volumes is None:
            try:
                volumes = []
                unit = 1
                for system in self._systems:
                    if system.atoms.lattice_vectors is not None:
                        cell = system.atoms.lattice_vectors.magnitude
                        unit = system.atoms.lattice_vectors.units
                        volumes.append(get_volume(cell))
                self.results.volumes = np.array(volumes) * unit**3
            except Exception:
                pass

        if not self.results.eos_fit:
            function_name_map = {
                'birch_murnaghan': 'birchmurnaghan',
                'pourier_tarantola': 'pouriertarantola',
                'vinet': 'vinet',
                'murnaghan': 'murnaghan',
                'birch_euler': 'birch',
            }
            if self.results.volumes is not None and self.results.energies is not None:
                # convert to ase units in order for function optimization to work
                volumes = self.results.volumes.to('angstrom ** 3').magnitude
                energies = self.results.energies.to('eV').magnitude
                for function_name, ase_name in function_name_map.items():
                    try:
                        eos = aseEOS(volumes, energies, ase_name)
                        eos.fit()
                        fitted_energies = eos.func(volumes, *eos.eos_parameters)
                        rms_error = np.sqrt(np.mean((fitted_energies - energies) ** 2))
                        eos_fit = EOSFit(
                            function_name=function_name,
                            fitted_energies=fitted_energies * ureg.eV,
                            bulk_modulus=eos.B * ureg.eV / ureg.angstrom**3,
                            equilibrium_volume=eos.v0 * ureg.angstrom**3,
                            equilibrium_energy=eos.e0 * ureg.eV,
                            rms_error=rms_error,
                        )
                        self.results.eos_fit.append(eos_fit)
                    except Exception:
                        logger.warning('EOS fit not succesful.')

    # @staticmethod
    # def archive_path_to_jmespath(path: str) -> str:
    #     """
    #     Converts an archive path like 'run/0/system/-1' to a jmespath like 'run[0].system[-1]'.
    #     """
    #     if not path:
    #         return ''
    #     parts = path.strip('/').split('/')
    #     jmes = []
    #     i = 0
    #     while i < len(parts):
    #         part = parts[i]
    #         # If next part is an integer, treat as index
    #         if i + 1 < len(parts) and parts[i + 1].lstrip('-').isdigit():
    #             jmes.append(f"{part}[{parts[i + 1]}]")
    #             i += 2
    #         else:
    #             jmes.append(part)
    #             i += 1
    #     return '.'.join(jmes)
