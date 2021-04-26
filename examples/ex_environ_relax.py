from aiida.orm.utils import load_node, load_code
from aiida.orm import Dict
from aiida.engine import submit
from aiida.orm.nodes.data.upf import get_pseudos_from_structure
from aiida.plugins.factories import WorkflowFactory

# try loading aiida-environ, everything stored as nodes already
code = load_code(109)
workchain = WorkflowFactory('environ.pw.relax')
builder = workchain.get_builder()
builder.metadata.label = "Environ test"
builder.metadata.description = "Test of environ plugin"
builder.base.pw.metadata.options.resources = {'num_machines': 1}
builder.base.pw.metadata.options.max_wallclock_seconds = 6 * 60 * 60
builder.base.pw.code = code

structure = load_node(993)
pp_water = get_pseudos_from_structure(structure, 'SSSPe')
kpoints = load_node(149)
parameters = load_node(148)
environ_parameters = {
    "ENVIRON": {},
    "BOUNDARY": {},
    "ELECTROSTATIC": {}
}

builder.structure = structure
builder.base.kpoints = kpoints
builder.base.pw.parameters = parameters
builder.base.pw.pseudos = pp_water
builder.base.pw.environ_parameters = Dict(dict=environ_parameters)
print(builder)

calculation = submit(builder)
