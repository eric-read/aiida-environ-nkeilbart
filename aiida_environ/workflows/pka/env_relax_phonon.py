# -*- coding: utf-8 -*-
"""
Workchain to perform pKa calculations using the Environ module
coupled with the Quantum ESPRESSO pw.x.
"""
from aiida import orm # type: ignore
from aiida.common import AttributeDict
from aiida.common.lang import type_check
from aiida.engine import ToContext, WorkChain
from aiida.plugins import WorkflowFactory
from aiida_quantumespresso.common.types import RelaxType
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida.orm import StructureData, to_aiida_type

PwRelaxWorkChain = WorkflowFactory("environ.pw.relax")
PhononWorkChain = WorkflowFactory("environ.pka.env_phonon")

def validate_inputs(inputs, _):
    """Validate the top level namespace."""
    parameters = inputs["vacuum"]["base"]["pw"]["parameters"].get_dict()

    if "relaxation_scheme" not in inputs and \
       "calculation" not in parameters.get("CONTROL", {}):
        return (
            "The parameters in `base.pw.parameters` do not specify the "
            "required key `CONTROL.calculation`."
        )

class EnvRelaxPhononWorkChain(WorkChain, ProtocolMixin):
    """
    Workchain to perform relax+Phonon calculations using Quantum ESPRESSO pw.x
    with Environ and Phonopy.
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.expose_inputs(
            PwRelaxWorkChain,
            namespace='scf',
            exclude=('clean_workdir'),
            namespace_options={
                'help': (
                    'Inputs for the `EnvPwRelaxWorkChain`'
                )
            }
        )
        spec.expose_inputs(
            PhononWorkChain,
            namespace='phonon',
            include=(
                'primitive_matrix',
                'symmetry',
                'supercell_matrix',
                'displacement_generator',
                'phonopy',
                'settings',
                'clean_workdir'
            ),
            namespace_options={
                'help': ('Inputs for the `EnvPhononWorkChain`')
            }
        )
        spec.input(
            "phonon.parallelization",
            valid_type=orm.Dict,
            required=False,
            serializer=to_aiida_type,
            help="Parallelization options to use in pw subprocesses in"
                 "phonon calculations."
        )
        spec.input(
            "phonon.options",
            valid_type=orm.Dict,
            required=False,
            serializer=to_aiida_type,
            help=(
                "Options for metadata to use in pw subprocesses in "
                "phonon calculations."
            )
        )            
        spec.input(
            'clean_workdir',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help=(
                'If `True`, work directories of all called calculations will '
                'be cleaned at the end of execution.'
            )
        )
        spec.input(
            'clean_phonon_workdir',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help=(
                'If `True`, work directories of phonon calculations'
                'will be cleaned at the end of execution.'
            )
        )
        spec.inputs.validator = validate_inputs
        spec.outline(
            cls.setup,

            cls.run_relax,
            cls.check_relax,

            cls.run_environ_phonon,
            cls.check_environ_phonon,
        )
        spec.expose_outputs(
            PwRelaxWorkChain,
            namespace='scf'
        )
        spec.expose_outputs(
            PhononWorkChain,
            namespace='phonon'
        )

        spec.exit_code(
            403,
            'ERROR_ENVIRON_RELAX_CALCULATION_FAILED',
            message='the environ PwRelaxWorkChain failed'
        )
        spec.exit_code(
            404,
            'ERROR_ENVIRON_PHONON_CALCULATION_FAILED',
            message='The solution phonon PhononWorkChain failed '
        )

    @classmethod
    def get_protocol_filepath(cls):
        """
        Return ``pathlib.Path`` to the ``.yaml`` file that defines 
        the protocols.
        """
        from aiida_quantumespresso.workflows.protocols import pw as pw_protocols
        from importlib_resources import files

        return files(pw_protocols) / "relax.yaml"

    @classmethod
    def get_builder_from_protocol(
            cls,
            code: orm.Code,
            phonopy_code: orm.Code,
            structure: StructureData,
            protocol: orm.Dict = None,
            overrides: orm.Dict = None,
            options=None,
            relax_type=RelaxType.POSITIONS,
            pseudo_family='SSSP/1.3/PBE/precision',
            clean_workdir: bool = False,
            clean_phonon_workdir: bool = False,
            **kwargs,
    ):
        """
        Return a builder prepopulated with inputs selected according to the 
        chosen protocol.

        :param code: 
            the ``Code`` instance configured for the ``quantumespresso.pw`` 
            plugin.
        :param structure: 
            the ``StructureData`` instance to use.
        :param protocol: 
            protocol to use, if not specified, the default will be used.
        :param overrides: 
            optional dictionary of inputs to override the defaults of the 
            protocol.
        :param options: 
            A dictionary of options that will be recursively set for the 
            ``metadata.options`` input of all the ``CalcJobs`` that are 
            nested in this work chain.
        :param relax_type: 
            the relax type to use: should be a value of the enum 
            ``common.types.RelaxType``.
        :param kwargs: 
            additional keyword arguments that will be passed to the 
            ``get_builder_from_protocol`` of all the sub processes 
            that are called by this workchain.

        :return: 
            a process builder instance with all inputs defined ready for 
            launch.
        """

        type_check(relax_type, RelaxType)

        args = (code, structure, protocol)
        builder = cls.get_builder()
        inputs = cls.get_protocol_inputs(protocol, overrides)
        scf = PwRelaxWorkChain.get_builder_from_protocol(
            *args,
            relax_type=relax_type,
            overrides=inputs.get('scf', None), 
            options=options,
            **kwargs
        )
        phonon = PhononWorkChain.get_builder_from_protocol(
            *args,
            phonopy_code=phonopy_code,
            options=options,
            **kwargs,
            clean_workdir=orm.Bool(True),
        )
        scf["base"]["pw"].pop("structure", None)
        scf.pop("clean_workdir", None)

        # Declare the environ.in file input
        environ_input = {
            'ENVIRON': {
                'env_static_permittivity': 78.3,
                'env_pressure': -0.35,
                'env_surface_tension': 50.0,
                'verbose': 1,
                'environ_thr': 100.0
            },
            'BOUNDARY': {
                'alpha': 1.12,
                'radius_mode': 'muff',
                'solvent_mode': 'ionic',
            },
            'ELECTROSTATIC': {
                'solver': 'cg',
                'tol': 1.E-11
            }
        }

        scf['base']['pw']['environ_parameters'] = orm.Dict(dict=environ_input)

        builder.code = code
        builder.phonopy_code = phonopy_code
        builder.scf = scf
        builder.phonon = phonon
        builder.structure = structure
        builder.clean_workdir = orm.Bool(clean_workdir)
        builder.clean_phonon_workdir = orm.Bool(clean_phonon_workdir)
        builder.pseudo_family = orm.Str(pseudo_family)
        return builder

    def setup(self):
        """Input validation and context setup."""
        self.ctx.pwrelax_input = AttributeDict(
            self.exposed_inputs(
                PwRelaxWorkChain,
            )
        )
        return

    def run_relax(self):
        """
        Run solution environment simulations for all structures.
        """
        inputs = AttributeDict(
            self.exposed_inputs(
                PwRelaxWorkChain,
                namespace='scf'
            )
        )

        # inputs.metadata.call_link_label = f'CALL'
        self.ctx.scf_inputs = inputs
        future = self.submit(PwRelaxWorkChain, **inputs)
        self.report(
            f'submitting solution `EnvPwRelaxWorkChain` <PK={future.pk}> '
            f'<UUID={future.uuid}>.'
        )
        self.to_context(**{f'scf': future})
        return


    def check_relax(self):
        """
        Inspect output of vacuum simulations.
        """
        workchain = self.ctx.scf

        if not workchain.is_finished_ok:
            self.report(
                f'Solution `EnvPwRelaxWorkChain` with <PK={workchain.pk}> <UUID={workchain.uuid} failed'
                f'with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_ENVIRON_RELAX_CALCULATION_FAILED
        else:
            self.report('Solution optimization finished')
            self.out_many(self.exposed_outputs(self.ctx.scf, PwRelaxWorkChain, namespace='scf', agglomerate=False))

        return

    def run_environ_phonon(self):
        scf_inputs = self.ctx.scf_inputs
        phonon_inputs = AttributeDict(
            self.exposed_inputs(
                PhononWorkChain,
                namespace='phonon'
            )
        )
        structure = self.ctx.scf.outputs.output_structure
        parameters = scf_inputs.base.pw.parameters.get_dict()
        parameters['CONTROL']['calculation'] = 'scf'
        parameters['CONTROL']['tprnfor'] = True
        parameters['CONTROL']['restart_mode'] = 'from_scratch'
        parameters['ELECTRONS']['startingpot'] = 'file'
        parameters['ELECTRONS']['startingwfc'] = 'file'
        parameters.pop('IONS', None)
        parameters.pop('CELL', None)
        scf_inputs.base.pw.parameters = parameters
        environ_parameters = scf_inputs.base.pw.environ_parameters.get_dict()
        environ_parameters['ENVIRON']['environ_restart'] = True
        scf_inputs.base.pw.environ_parameters = environ_parameters
        scf_inputs.base.pw.parent_folder = \
            self.ctx.scf.outputs.remote_folder
        phonon_inputs.scf = scf_inputs.base
        phonon_inputs.scf.pw.structure = structure
        # phonon_inputs.metadata.call_link_label = f'CALL'
        phonon_inputs.clean_workdir = self.inputs.clean_workdir \
            or self.inputs.clean_phonon_workdir
        if 'parallelization' in self.inputs.phonon:
            phonon_inputs.scf.pw.parallelization = \
                self.inputs.phonon.parallelization.get_dict()
        if 'options' in self.inputs.phonon:
            phonon_inputs.scf.pw.metadata.options = \
                self.inputs.phonon.options.get_dict()

        future = self.submit(PhononWorkChain, **phonon_inputs)
        self.report((
            f'submitting `EnvPhononWorkChain` in solution <PK={future.pk}> '
            f'<UUID={future.uuid}>.'
        ))
        self.to_context(**{f'phonon': future})
        return

    def check_environ_phonon(self):
        workchain = self.ctx.phonon
        if not workchain.is_finished_ok:
            self.report((
                f'Solution `EnvPhononWorkChain` with <PK={workchain.pk}> '
                f'<UUID={workchain.uuid} failed with exit status '
                f'{workchain.exit_status}'
            ))
            return self.exit_codes.ERROR_ENVIRON_PHONON_CALCULATION_FAILED
        else:
            self.report('Solution phonon calculation finished')
            self.out_many(
                self.exposed_outputs(
                    self.ctx.phonon,
                    PhononWorkChain, 
                    namespace='phonon',
                    agglomerate=False
                    )
                )
        return

    def on_terminated(self):
        """
        Clean the working directories of all child calculations if 
        `clean_workdir=True` in the inputs.
        """
        super().on_terminated()
        if self.inputs.clean_workdir.value is False:
            self.report("remote folders will not be cleaned")
            return

        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (IOError, OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report((
                f"cleaned remote folders of calculations: "
                f"{' '.join(map(str, cleaned_calcs))}"
            ))