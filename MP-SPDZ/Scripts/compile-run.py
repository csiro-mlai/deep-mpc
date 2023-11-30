#!/usr/bin/env python3

import os, sys

sys.path.insert(0, os.path.dirname(sys.argv[0]) + '/..')

from Compiler.compilerLib import Compiler

try:
    split = sys.argv.index('--')
except ValueError:
    split = len(sys.argv)

compiler_args = sys.argv[1:split]
runtime_args = sys.argv[split + 1:]
compiler = Compiler(execute=True, custom_args=compiler_args)
compiler.prep_compile()
prog = compiler.compile_file()

if prog.options.hostfile:
    compiler.remote_execution(runtime_args)
else:
    compiler.local_execution(runtime_args)
