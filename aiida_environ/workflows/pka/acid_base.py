# -*- coding: utf-8 -*-
"""
Workchain to perform relaxations and phonon calculations on conjugate acid base pairs
"""
from aiida import orm
from aiida.common import AttributeDict
from aiida.common.lang import type_check
from aiida.engine import ToContext, WorkChain
from aiida.plugins import WorkflowFactory
from aiida_quantumespresso.common.types import RelaxType
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida.orm import StructureData

EnvRelaxPhononWorkChain = WorkflowFactory("environ.pka.env_relax_phonon")

class AcidBaseWorkChain(WorkChain, ProtocolMixin):
    """
    Workchain to perform pKa calculations using Quantum ESPRESSO pw.x and Phonopy
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)

        spec.expose_inputs(
            EnvRelaxPhononWorkChain,
            namespace='acid',
            exclude=('clean_workdir', 'clean_phonon_workdir'),
            namespace_options={
                'help': ('Inputs for the `EnvRelaxPhononWorkChain` calculation for the acid')
            }
        )
        spec.expose_inputs(
            EnvRelaxPhononWorkChain,
            namespace='base',
            exclude=('clean_workdir', 'clean_phonon_workdir'),
            namespace_options={
                'help': ('Inputs for the `EnvRelaxPhononWorkChain` calculation for the base')
            }
        )
        spec.input(
            'clean_workdir',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help=('If `True`, work directories of all called calculation '
                  'will be cleaned at the end of execution.')
        )
        spec.input(
            'clean_phonon_workdir',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help=('If `True`, work directories of all called calculations in Phonon calculations'
                  'will be cleaned at the end of execution.')
        )

        spec.outline(
            cls.setup,
            cls.run_acid,
            cls.check_acid,
            cls.run_base,
            cls.check_base,
            cls.results,
        )

        spec.exit_code(
            401,
            'ERROR_ACID_CALCULATION_FAILED',
            message='the acid `EnvRelaxPhononWorkChain` calculation failed'
        )
        spec.exit_code(
            402,
            'ERROR_BASE_CALCULATION_FAILED',
            message='the base `EnvRelaxPhononWorkChain` calculation failed'
        )

        spec.expose_outputs(
            EnvRelaxPhononWorkChain,
            namespace='acid',
        )
        spec.expose_outputs(
            EnvRelaxPhononWorkChain,
            namespace='base',
        )

    @classmethod
    def get_builder_from_protocol(
            cls,
            code: orm.Code,
            phonopy_code: orm.Code,
            acid_structure: StructureData,
            base_structure: StructureData,
            protocol: orm.Dict = None,
            overrides: orm.Dict = None,
            options=None,
            relax_type=RelaxType.POSITIONS,
            pseudo_family='SSSP/1.3/PBE/precision',
            clean_workdir: bool = False,
            clean_phonon_workdir: bool = True,
            **kwargs,
    ):
        """
        Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param structure: the ``StructureData`` instance to use.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param options: A dictionary of options that will be recursively set for the ``metadata.options`` input of all
            the ``CalcJobs`` that are nested in this work chain.
        :param relax_type: the relax type to use: should be a value of the enum ``common.types.RelaxType``.
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all the
            sub processes that are called by this workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """

        type_check(relax_type, RelaxType)

        args = (code, phonopy_code, protocol, overrides, options, relax_type, pseudo_family, clean_workdir, clean_phonon_workdir)
        builder = cls.get_builder()
        inputs = cls.get_protocol_inputs(protocol, overrides)
        acid_inputs = EnvRelaxPhononWorkChain.get_builder_from_protocol(
            code=code,
            phonopy_code=phonopy_code,
            structure=acid_structure,
            protocol=protocol,
            overrides=overrides,
            options=options,
            relax_type=relax_type,
            pseudo_family=pseudo_family,
            clean_workdir=clean_workdir,
            clean_phonon_workdir=clean_phonon_workdir,
            **kwargs
        )
        base_inputs = EnvRelaxPhononWorkChain.get_builder_from_protocol(
            code=code,
            phonopy_code=phonopy_code,
            structure=base_structure,
            protocol=protocol,
            overrides=overrides,
            options=options,
            relax_type=relax_type,
            pseudo_family=pseudo_family,
            clean_workdir=clean_workdir,
            clean_phonon_workdir=clean_phonon_workdir,
            **kwargs
        )
        builder.acid = acid_inputs
        builder.base = base_inputs
        builder.clean_workdir = orm.Bool(clean_workdir)
        builder.clean_phonon_workdir = orm.Bool(clean_phonon_workdir)
        return builder

    def setup(self):
        """Input validation and context setup."""
        self.ctx.acid_failed = True
        self.ctx.base_failed = True
        return

    def run_acid(self):
        inputs = AttributeDict(
            self.exposed_inputs(
                EnvRelaxPhononWorkChain,
                namespace='acid',
            )
        )
        inputs.clean_phonon_workdir = self.inputs.clean_phonon_workdir
        # inputs.metadata.call_link_label = f'acid_relax_and_Phonon'
        future = self.submit(EnvRelaxPhononWorkChain, **inputs)
        self.report(f'submitting acid `EnvRelaxPhononWorkChain` <PK={future.pk}> <UUID={future.uuid}>.')
        self.to_context(**{f'acid': future})
        return

    def check_acid(self):
        """
                Inspect output of acid simulations.
        """
        workchain = self.ctx.acid

        if not workchain.is_finished_ok:
            self.report(
                f'Acid `EnvRelaxPhononWorkChain` with <PK={workchain.pk}> <UUID={workchain.uuid} failed '
                f'with exit status {workchain.exit_status}'
            )
            return self.exit_codes.ERROR_ACID_CALCULATION_FAILED
        else:
            self.report('Acid Relaxation and Phonon workchain is finished.')
            self.out_many(
                self.exposed_outputs(self.ctx.acid, EnvRelaxPhononWorkChain, namespace='acid', agglomerate=False)
            )
        return

    def run_base(self):
        inputs = AttributeDict(
            self.exposed_inputs(
                EnvRelaxPhononWorkChain,
                namespace='base',
            )
        )
        inputs.clean_phonon_workdir = self.inputs.clean_phonon_workdir
        # inputs.metadata.call_link_label = f'base_relax_and_phonon'
        future = self.submit(EnvRelaxPhononWorkChain, **inputs)
        self.report(f'submitting base `EnvRelaxPhononWorkChain` <PK={future.pk}> <UUID={future.uuid}>.')
        self.to_context(**{f'base': future})
        return

    def check_base(self):
        """
        Inspect output of base simulations.
        """
        workchain = self.ctx.base

        if not workchain.is_finished_ok:
            self.report(
                f'Base `EnvRelaxPhononWorkChain` with <PK={workchain.pk}> <UUID={workchain.uuid} failed '
                f'with exit status {workchain.exit_status}'
            )
            return self.exit_codes.ERROR_BASE_CALCULATION_FAILED
        else:
            self.report('Base Relaxation and Phonon workchain is finished.')
            self.out_many(
                self.exposed_outputs(self.ctx.base, EnvRelaxPhononWorkChain, namespace='base', agglomerate=False)
            )
        return

    def results(self):
        return

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()
        
        if self.inputs.clean_phonon_workdir.value is True:
            self.report("remote folders for Phonon Calculations will be cleaned")
            return
        elif self.inputs.clean_workdir.value is False:
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
            self.report(
                f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}"
            )