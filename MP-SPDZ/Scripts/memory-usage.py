#!/usr/bin/env python3

import sys, os
import collections

sys.path.append('.')

from Compiler.program import *
from Compiler.instructions_base import *

if len(sys.argv) <= 1:
    print('Usage: %s <program>' % sys.argv[0])

res = collections.defaultdict(lambda: 0)
regs = collections.defaultdict(lambda: 0)
thread_regs = collections.defaultdict(lambda: 0)

def process(tapename, res, regs):
    for inst in Tape.read_instructions(tapename):
        t = inst.type
        if issubclass(t, DirectMemoryInstruction):
            res[type(inst.args[0])] = max(inst.args[1].i + inst.size,
                                       res[type(inst.args[0])]) + 1
        for arg in inst.args:
            if isinstance(arg, RegisterArgFormat):
                regs[type(arg)] = max(regs[type(arg)], arg.i + inst.size)

tapes = Program.read_tapes(sys.argv[1])

process(next(tapes), res, regs)

for tapename in tapes:
    process(tapename, res, thread_regs)

reverse_formats = dict((v, k) for k, v in ArgFormats.items())

regout = lambda regs: dict((reverse_formats[t], n) for t, n in regs.items())

def output(data):
    for t, n in data.items():
        if n:
            try:
                print('%10d %s' % (n, ArgFormats[t.removesuffix('w')].name))
            except:
                pass

total = 0
for x in res, regs, thread_regs:
    total += sum(x.values())

print ('Memory:')
output(regout(res))

print ('Registers in main thread:')
output(regout(regs))

if thread_regs:
    print ('Registers in other threads:')
    output(regout(thread_regs))

print ('The program requires at the very least %f GB of RAM per party.' % \
       (total * 8e-9))
