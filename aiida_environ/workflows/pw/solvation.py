# -*- coding: utf-8 -*-
from aiida.common import AttributeDict
from aiida.engine import if_, ToContext, WorkChain, calcfunction
from aiida.orm import Dict, Float, Bool, CalcJobNode
from aiida.plugins import WorkflowFactory
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin, recursive_merge

EnvPwBaseWorkChain = WorkflowFactory("environ.pw.base")


@calcfunction
def subtract_energy(x, y):
    return x - y


class PwSolvationWorkChain(WorkChain, ProtocolMixin):
    """
    WorkChain to compute the solvation energy for a given structure using 
    Quantum ESPRESSO pw.x + ENVIRON

    Expects one of two possible inputs by the user.
    1) An environ-parameter dictionary as per a regular environ calculation.
    2) An environ-parameter dictionary with shared variables and one/two 
       dictionaries for custom vacuum/solution input.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(
            EnvPwBaseWorkChain,
            namespace="base",
            exclude=('clean_workdir'),
            namespace_options={
                "help": "General inputs for the `EnvPwBaseCalculation`."
            },
        )
        spec.input(
            "environ_vacuum",
            valid_type=Dict,
            required=False,
            help="The environ inputs to run an environ simulation in vacuum",
        )
        spec.input(
            "environ_solution",
            valid_type=Dict,
            required=False,
            help="The environ inputs to run an environ simulation in solution",
        )
        spec.input(
            "energy_vacuum",
            valid_type=Float,
            required=False,
            help=(
                "The vacuum energy in eV, if provided, skips the vacuum "
                "calculation"),
        )
        spec.input(
            'use_vacuum_output_structure',
            valid_type=Bool,
            default=lambda: Bool(True),
            required=False,
            help=(
                "Use the relaxed geometry from the environ simulation in "
                "vacuum as the initial structure in the solution "
                "calculation"),
        )
        spec.input(
            'clean_workdir',
            valid_type=Bool,
            default=lambda: Bool(True),
            required=False,
            help=(
                "If `True`, work directories of all called calculations will be cleaned at the end of execution."
            )
        )
        spec.outline(
            if_(cls.should_run_vacuum)(
                cls.vacuum_setup,
                cls.run_vacuum,
                cls.inspect_vacuum,
            ),
            cls.setup_solution,
            cls.run_solution,
            cls.inspect_solution,
            cls.results,
        )
        spec.output("solvation_energy", valid_type=Float)

        spec.expose_outputs(
            EnvPwBaseWorkChain,
            namespace='vacuum',
            namespace_options={'required': False},
        )

        spec.expose_outputs(
            EnvPwBaseWorkChain,
            namespace='solution',
        )

        spec.exit_code(
            400,'ERROR_VACUUM_FAILED',
            message='Vacuum Calculation Failed'
        )
        spec.exit_code(
            401,'ERROR_SOLUTION_FAILED',
            message='Solution Calculation Failed'
        )
    # @classmethod
    # def get_builder_from_protocol(
    #     cls,
    #     code,
    #     structure,
    #     protocol=None,
    #     overrides=None
    # ):
    #     """Return a builder prepopulated with inputs selected according to the chosen protocol.
    #     # TODO add protocols for solvation, requires inherit ProtocolMixin

    #     :param code: the ``Code`` instance configured for the ``environ.pw`` plugin.
    #     :param structure: the ``StructureData`` instance to use.
    #     :param protocol: protocol to use, if not specified, the default will be used.
    #     :param overrides: optional dictionary of inputs to override the defaults of the protocol.
    #     """
    #     inputs = cls.get_protocol_inputs(protocol, overrides)

    #     args = (code, structure, protocol)
    #     base = EnvPwBaseWorkChain.get_builder_from_protocol(*args, overrides=inputs.get('base', None))

    #     builder = cls.get_builder()
    #     builder.base = base

    #     return builder

    def vacuum_setup(self):
        from copy import deepcopy

        self.ctx.vacuum_inputs = AttributeDict(
            self.exposed_inputs(EnvPwBaseWorkChain, namespace="base")
        )
        environ_parameters = self.inputs.base.pw.environ_parameters.get_dict()
        if "environ_vacuum" in self.inputs:
            vacuum_overrides = self.inputs.environ_vacuum.get_dict()
        else:
            vacuum_overrides = {}
        self.ctx.vacuum_inputs.pw.environ_parameters = deepcopy(
            recursive_merge(environ_parameters, vacuum_overrides)
        )
        # Set all the defaults

        self.ctx.vacuum_inputs.pw.environ_parameters.setdefault("ENVIRON", {})
        self.ctx.vacuum_inputs.pw.environ_parameters["ENVIRON"].setdefault("verbose", 0)
        self.ctx.vacuum_inputs.pw.environ_parameters["ENVIRON"].setdefault(
            "environ_thr", 1e-1
        )
        self.ctx.vacuum_inputs.pw.environ_parameters["ENVIRON"].setdefault(
            "environ_type", "vacuum"
        )
        self.ctx.vacuum_inputs.pw.environ_parameters["ENVIRON"].setdefault(
            "environ_restart", False
        )
        self.ctx.vacuum_inputs.pw.environ_parameters["ENVIRON"].setdefault(
            "env_electrostatic", True
        )

        self.ctx.vacuum_inputs.pw.environ_parameters.setdefault("ELECTROSTATIC", {})
        self.ctx.vacuum_inputs.pw.environ_parameters["ELECTROSTATIC"].setdefault(
            "solver", "direct"
        )
        self.ctx.vacuum_inputs.pw.environ_parameters["ELECTROSTATIC"].setdefault(
            "auxiliary", "none"
        )

        return

    def run_vacuum(self):
        future = self.submit(EnvPwBaseWorkChain, **self.ctx.vacuum_inputs)
        self.report(f"submitting vacuum `EnvPwBaseWorkChain` <PK={future.pk}> <UUID={future.uuid}>")
        return self.to_context(**{f'vacuum_workchain': future})

    def inspect_vacuum(self):
        workchain = self.ctx.vacuum_workchain
        if not workchain.is_finished_ok:
            self.report(
                f'Vacuum `EnvPwBaseWorkChain` failed with exit status {workchain.exit_status}'
            )
            return self.exit_codes.ERROR_VACUUM_FAILED
        else:
            self.report(f'Vacuum `EnvPwBaseWorkChain` succeeded')
            self.out_many(
                self.exposed_outputs(workchain, EnvPwBaseWorkChain, namespace='vacuum', agglomerate=False)
            )
        return

    def should_run_vacuum(self):
        return  "energy_vacuum" not in self.inputs

    def setup_solution(self):
        from copy import deepcopy

        self.ctx.solution_inputs = AttributeDict(
            self.exposed_inputs(EnvPwBaseWorkChain, namespace="base")
        )
        environ_parameters = self.inputs.base.pw.environ_parameters.get_dict()
        if "environ_solution" in self.inputs:
            solution_overrides = self.inputs.environ_solution.get_dict()
        else:
            solution_overrides = {}
        parameters = self.ctx.solution_inputs.pw.parameters.get_dict()
        parameters['CONTROL']['restart_mode'] = 'from_scratch'
        calculation_type = parameters['CONTROL']['calculation']
        if calculation_type == 'scf':
            #Use potential and wfc from vacuum calculation to speed up solution calculation
            self.ctx.solution_inputs.pw.parent_folder = self.ctx.vacuum_workchain.outputs.remote_folder
            parameters['ELECTRONS']['startingpot'] = 'file'
            parameters['ELECTRONS']['startingwfc'] = 'file'
            solution_overrides['ENVIRON']['environ_restart'] = True
        elif calculation_type in ['relax', 'vc', 'vc-relax'] and self.inputs.use_vacuum_output_structure:
            #Use potential and wfc from vacuum calculation to speed up solution calculation
            self.ctx.solution_inputs.pw.structure = self.ctx.vacuum_workchain.outputs.output_structure
            self.ctx.solution_inputs.pw.parent_folder = self.ctx.vacuum_workchain.outputs.remote_folder
            parameters['ELECTRONS']['startingpot'] = 'file'
            parameters['ELECTRONS']['startingwfc'] = 'file'
            solution_overrides['ENVIRON']['environ_restart'] = True
        
        self.ctx.solution_inputs.pw.parameters = parameters
        
        #Merge base environ_parameters and solution overrides
        self.ctx.solution_inputs.pw.environ_parameters = deepcopy(
            recursive_merge(environ_parameters, solution_overrides)
        )
        #self.ctx.solution_inputs.pw.environ_parameters.setdefault("ENVIRON", {})
        #self.ctx.solution_inputs.pw.environ_parameters['ENVIRON'].setdefault(
        #        "verbose", 0
        #)
        #self.ctx.solution_inputs.pw.environ_parameters['ENVIRON'].setdefault(
        #        "environ_restart", False
        #)
        #self.ctx.solution_inputs.pw.environ_parameters['ENVIRON'].setdefault(
        #        "env_electrostatic", True
        #)
        #self.ctx.solution_inputs.pw.environ_parameters.setdefault("ELECTROSTATIC", {})
        #self.ctx.solution_inputs.pw.environ_parameters.setdefault(
        #        "solver", 'cg'
        #)

        return
    
    def run_solution(self):
        future = self.submit(EnvPwBaseWorkChain, **self.ctx.solution_inputs)
        self.report(f'submitting solution `EnvPwBaseWorkChain` <PK={future.pk}> <UUID={future.uuid}>')
        return self.to_context(**{'solution_workchain': future})

    def inspect_solution(self):
        workchain = self.ctx.solution_workchain
        if not workchain.is_finished_ok:
            self.report(
                f'Solution `EnvPwBaseWorkChain` failed with exit status {workchain.exit_status}'
            )
            return self.exit_codes.ERROR_SOLUTION_FAILED
        else:
            self.report(f'Solution `EnvPwBaseWorkChain` succeeded')
            self.out_many(
                self.exposed_outputs(workchain, EnvPwBaseWorkChain, namespace='solution', agglomerate=False)
            )
        return

    def results(self):
        # subtract energy in water calculation by energy in vacuum calculation
        if "energy_vacuum" in self.inputs:
            e_vacuum = self.inputs.energy_vacuum
        else:
            e_vacuum = self.ctx.vacuum_workchain.outputs.output_parameters["energy"]

        e_solvent = self.ctx.solution_workchain.outputs.output_parameters["energy"]
        self.ctx.energy_difference = subtract_energy(Float(e_solvent), Float(e_vacuum))
        self.out("solvation_energy", self.ctx.energy_difference)
        return

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()
        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be clean')
            return
        
        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean() # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (IOError, OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report(f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}")

