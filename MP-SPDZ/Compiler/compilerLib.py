import inspect
import os
import re
import sys
import tempfile
import subprocess
from optparse import OptionParser

from Compiler.exceptions import CompilerError

from .GC import types as GC_types
from .program import Program, defaults


class Compiler:
    def __init__(self, custom_args=None, usage=None, execute=False):
        if usage:
            self.usage = usage
        else:
            self.usage = "usage: %prog [options] filename [args]"
        self.execute = execute
        self.custom_args = custom_args
        self.build_option_parser()
        self.VARS = {}
        self.root = os.path.dirname(__file__) + '/..'

    def build_option_parser(self):
        parser = OptionParser(usage=self.usage)
        parser.add_option(
            "-n",
            "--nomerge",
            action="store_false",
            dest="merge_opens",
            default=defaults.merge_opens,
            help="don't attempt to merge open instructions",
        )
        parser.add_option("-o", "--output", dest="outfile", help="specify output file")
        parser.add_option(
            "-a",
            "--asm-output",
            dest="asmoutfile",
            help="asm output file for debugging",
        )
        parser.add_option(
            "-g",
            "--galoissize",
            dest="galois",
            default=defaults.galois,
            help="bit length of Galois field",
        )
        parser.add_option(
            "-d",
            "--debug",
            action="store_true",
            dest="debug",
            help="keep track of trace for debugging",
        )
        parser.add_option(
            "-c",
            "--comparison",
            dest="comparison",
            default="log",
            help="comparison variant: log|plain|inv|sinv",
        )
        parser.add_option(
            "-M",
            "--preserve-mem-order",
            action="store_true",
            dest="preserve_mem_order",
            default=defaults.preserve_mem_order,
            help="preserve order of memory instructions; possible efficiency loss",
        )
        parser.add_option(
            "-O",
            "--optimize-hard",
            action="store_true",
            dest="optimize_hard",
            help="lower number of rounds at higher compilation cost "
            "(disables -C and increases the budget to 100000)",
        )
        parser.add_option(
            "-u",
            "--noreallocate",
            action="store_true",
            dest="noreallocate",
            default=defaults.noreallocate,
            help="don't reallocate",
        )
        parser.add_option(
            "-m",
            "--max-parallel-open",
            dest="max_parallel_open",
            default=defaults.max_parallel_open,
            help="restrict number of parallel opens",
        )
        parser.add_option(
            "-D",
            "--dead-code-elimination",
            action="store_true",
            dest="dead_code_elimination",
            default=defaults.dead_code_elimination,
            help="eliminate instructions with unused result",
        )
        parser.add_option(
            "-p",
            "--profile",
            action="store_true",
            dest="profile",
            help="profile compilation",
        )
        parser.add_option(
            "-s",
            "--stop",
            action="store_true",
            dest="stop",
            help="stop on register errors",
        )
        parser.add_option(
            "-R",
            "--ring",
            dest="ring",
            default=defaults.ring,
            help="bit length of ring (default: 0 for field)",
        )
        parser.add_option(
            "-B",
            "--binary",
            dest="binary",
            default=defaults.binary,
            help="bit length of sint in binary circuit (default: 0 for arithmetic)",
        )
        parser.add_option(
            "-G",
            "--garbled-circuit",
            dest="garbled",
            action="store_true",
            help="compile for binary circuits only (default: false)",
        )
        parser.add_option(
            "-F",
            "--field",
            dest="field",
            default=defaults.field,
            help="bit length of sint modulo prime (default: 64)",
        )
        parser.add_option(
            "-P",
            "--prime",
            dest="prime",
            default=defaults.prime,
            help="prime modulus (default: not specified)",
        )
        parser.add_option(
            "-I",
            "--insecure",
            action="store_true",
            dest="insecure",
            help="activate insecure functionality for benchmarking",
        )
        parser.add_option(
            "-b",
            "--budget",
            dest="budget",
            help="set budget for optimized loop unrolling (default: %d)" % \
            defaults.budget,
        )
        parser.add_option(
            "-X",
            "--mixed",
            action="store_true",
            dest="mixed",
            help="mixing arithmetic and binary computation",
        )
        parser.add_option(
            "-Y",
            "--edabit",
            action="store_true",
            dest="edabit",
            help="mixing arithmetic and binary computation using edaBits",
        )
        parser.add_option(
            "-Z",
            "--split",
            default=defaults.split,
            dest="split",
            help="mixing arithmetic and binary computation "
            "using direct conversion if supported "
            "(number of parties as argument)",
        )
        parser.add_option(
            "--invperm",
            action="store_true",
            dest="invperm",
            help="speedup inverse permutation (only use in two-party, "
            "semi-honest environment)"
        )
        parser.add_option(
            "-C",
            "--CISC",
            action="store_true",
            dest="cisc",
            help="faster CISC compilation mode "
            "(used by default unless -O is given)",
        )
        parser.add_option(
            "-K",
            "--keep-cisc",
            dest="keep_cisc",
            help="don't translate CISC instructions",
        )
        parser.add_option(
            "-l",
            "--flow-optimization",
            action="store_true",
            dest="flow_optimization",
            help="optimize control flow",
        )
        parser.add_option(
            "-v",
            "--verbose",
            action="store_true",
            dest="verbose",
            help="more verbose output",
        )
        if self.execute:
            parser.add_option(
                "-E",
                "--execute",
                dest="execute",
                help="protocol to execute with",
            )
            parser.add_option(
                "-H",
                "--hostfile",
                dest="hostfile",
                help="hosts to execute with",
            )
        self.parser = parser

    def parse_args(self):
        self.options, self.args = self.parser.parse_args(self.custom_args)
        if self.execute:
            if not self.options.execute:
                raise CompilerError("must give name of protocol with '-E'")
            protocol = self.options.execute
            if protocol.find("ring") >= 0 or protocol.find("2k") >= 0 or \
               protocol.find("brain") >= 0 or protocol == "emulate":
                if not (self.options.ring or self.options.binary):
                    self.options.ring = "64"
                if self.options.field:
                    raise CompilerError(
                        "field option not compatible with %s" % protocol)
            else:
                if protocol.find("bin") >= 0 or  protocol.find("ccd") >= 0 or \
                   protocol.find("bmr") >= 0 or \
                   protocol in ("replicated", "tinier", "tiny", "yao"):
                    if not self.options.binary:
                        self.options.binary = "32"
                    if self.options.ring or self.options.field:
                        raise CompilerError(
                            "ring/field options not compatible with %s" %
                            protocol)
                if self.options.ring:
                    raise CompilerError(
                        "ring option not compatible with %s" % protocol)
            if protocol == "emulate":
                self.options.keep_cisc = ''

    def build_program(self, name=None):
        self.prog = Program(self.args, self.options, name=name)
        if self.execute:
            if self.options.execute in \
               ("emulate", "ring", "rep-field", "rep4-ring"):
                self.prog.use_trunc_pr = True
            if self.options.execute in ("ring", "ps-rep-ring", "sy-rep-ring"):
                self.prog.use_split(3)
            if self.options.execute in ("semi2k",):
                self.prog.use_split(2)
            if self.options.execute in ("rep4-ring",):
                self.prog.use_split(4)

    def build_vars(self):
        from . import comparison, floatingpoint, instructions, library, types

        # add all instructions to the program VARS dictionary
        instr_classes = [
            t[1] for t in inspect.getmembers(instructions, inspect.isclass)
        ]

        for mod in (types, GC_types):
            instr_classes += [
                t[1]
                for t in inspect.getmembers(mod, inspect.isclass)
                if t[1].__module__ == mod.__name__
            ]

        instr_classes += [
            t[1]
            for t in inspect.getmembers(library, inspect.isfunction)
            if t[1].__module__ == library.__name__
        ]

        for op in instr_classes:
            self.VARS[op.__name__] = op

        # backward compatibility for deprecated classes
        self.VARS["sbitint"] = GC_types.sbitintvec
        self.VARS["sbitfix"] = GC_types.sbitfixvec

        # add open and input separately due to name conflict
        self.VARS["vopen"] = instructions.vasm_open
        self.VARS["gopen"] = instructions.gasm_open
        self.VARS["vgopen"] = instructions.vgasm_open
        self.VARS["ginput"] = instructions.gasm_input

        self.VARS["comparison"] = comparison
        self.VARS["floatingpoint"] = floatingpoint

        self.VARS["program"] = self.prog
        if self.options.binary:
            self.VARS["sint"] = GC_types.sbitintvec.get_type(int(self.options.binary))
            self.VARS["sfix"] = GC_types.sbitfixvec
            for i in [
                "cint",
                "cfix",
                "cgf2n",
                "sintbit",
                "sgf2n",
                "sgf2nint",
                "sgf2nuint",
                "sgf2nuint32",
                "sgf2nfloat",
                "cfloat",
                "squant",
            ]:
                del self.VARS[i]

    def prep_compile(self, name=None, build=True):
        self.parse_args()
        if len(self.args) < 1 and name is None:
            self.parser.print_help()
            exit(1)
        if build:
            self.build(name=name)

    def build(self, name=None):
        self.build_program(name=name)
        self.build_vars()

    def compile_file(self):
        """Compile a file and output a Program object.

        If options.merge_opens is set to True, will attempt to merge any
        parallelisable open instructions."""
        print("Compiling file", self.prog.infile)

        with open(self.prog.infile, "r") as f:
            changed = False
            if self.options.flow_optimization:
                output = []
                if_stack = []
                for line in f:
                    if if_stack and not re.match(if_stack[-1][0], line):
                        if_stack.pop()
                    m = re.match(
                        r"(\s*)for +([a-zA-Z_]+) +in " r"+range\(([0-9a-zA-Z_.]+)\):",
                        line,
                    )
                    if m:
                        output.append(
                            "%s@for_range_opt(%s)\n" % (m.group(1), m.group(3))
                        )
                        output.append("%sdef _(%s):\n" % (m.group(1), m.group(2)))
                        changed = True
                        continue
                    m = re.match(r"(\s*)if(\W.*):", line)
                    if m:
                        if_stack.append((m.group(1), len(output)))
                        output.append("%s@if_(%s)\n" % (m.group(1), m.group(2)))
                        output.append("%sdef _():\n" % (m.group(1)))
                        changed = True
                        continue
                    m = re.match(r"(\s*)elif\s+", line)
                    if m:
                        raise CompilerError("elif not supported")
                    if if_stack:
                        m = re.match("%selse:" % if_stack[-1][0], line)
                        if m:
                            start = if_stack[-1][1]
                            ws = if_stack[-1][0]
                            output[start] = re.sub(
                                r"^%s@if_\(" % ws, r"%s@if_e(" % ws, output[start]
                            )
                            output.append("%s@else_\n" % ws)
                            output.append("%sdef _():\n" % ws)
                            continue
                    output.append(line)
                if changed:
                    infile = tempfile.NamedTemporaryFile("w+", delete=False)
                    for line in output:
                        infile.write(line)
                    infile.seek(0)
                else:
                    infile = open(self.prog.infile)
            else:
                infile = open(self.prog.infile)

        # make compiler modules directly accessible
        sys.path.insert(0, "%s/Compiler" % self.root)
        # create the tapes
        exec(compile(infile.read(), infile.name, "exec"), self.VARS)

        if changed and not self.options.debug:
            os.unlink(infile.name)

        return self.finalize_compile()

    def register_function(self, name=None):
        """
        To register a function to be compiled, use this as a decorator.
        Example:

            @compiler.register_function('test-mpc')
            def test_mpc(compiler):
                ...
        """

        def inner(func):
            self.compile_name = name or func.__name__
            self.compile_function = func
            return func

        return inner

    def compile_func(self):
        if not (hasattr(self, "compile_name") and hasattr(self, "compile_func")):
            raise CompilerError(
                "No function to compile. "
                "Did you decorate a function with @register_fuction(name)?"
            )
        self.prep_compile(self.compile_name)
        print(
            "Compiling: {} from {}".format(self.compile_name, self.compile_func.__name__)
        )
        self.compile_function()
        self.finalize_compile()

    def finalize_compile(self):
        self.prog.finalize()

        if self.prog.req_num:
            print("Program requires at most:")
            for x in self.prog.req_num.pretty():
                print(x)

        if self.prog.verbose:
            print("Program requires:", repr(self.prog.req_num))
            print("Cost:", 0 if self.prog.req_num is None else self.prog.req_num.cost())
            print("Memory size:", dict(self.prog.allocated_mem))

        return self.prog

    @staticmethod
    def executable_from_protocol(protocol):
        match = {
            "ring": "replicated-ring",
            "rep-field": "replicated-field",
            "replicated": "replicated-bin"
        }
        if protocol in match:
            protocol = match[protocol]
        if protocol.find("bmr") == -1:
            protocol = re.sub("^mal-", "malicious-", protocol)
        if protocol == "emulate":
            return protocol + ".x"
        else:
            return protocol + "-party.x"

    def local_execution(self, args=[]):
        executable = self.executable_from_protocol(self.options.execute)
        if not os.path.exists("%s/%s" % (self.root, executable)):
            print("Creating binary for virtual machine...")
            try:
                subprocess.run(["make", executable], check=True, cwd=self.root)
            except:
                raise CompilerError(
                    "Cannot produce %s. " % executable + \
                    "Note that compilation requires a few GB of RAM.")
        vm = "%s/Scripts/%s.sh" % (self.root, self.options.execute)
        sys.stdout.flush()
        os.execl(vm, vm, self.prog.name, *args)

    def remote_execution(self, args=[]):
        vm = self.executable_from_protocol(self.options.execute)
        hosts = list(x.strip()
                     for x in filter(None, open(self.options.hostfile)))
        # test availability before compilation
        from fabric import Connection
        import subprocess
        print("Creating static binary for virtual machine...")
        subprocess.run(["make", "static/%s" % vm], check=True, cwd=self.root)

        # transfer files
        import glob
        hostnames = []
        destinations = []
        for host in hosts:
            split = host.split('/', maxsplit=1)
            hostnames.append(split[0])
            if len(split) > 1:
                destinations.append(split[1])
            else:
                destinations.append('.')
        connections = [Connection(hostname) for hostname in hostnames]
        print("Setting up players...")

        def run(i):
            dest = destinations[i]
            connection = connections[i]
            connection.run(
                "mkdir -p %s/{Player-Data,Programs/{Bytecode,Schedules}} " % \
                dest)
            # executable
            connection.put("%s/static/%s" % (self.root, vm), dest)
            # program
            dest += "/"
            connection.put("Programs/Schedules/%s.sch" % self.prog.name,
                           dest + "Programs/Schedules")
            for filename in glob.glob(
                    "Programs/Bytecode/%s-*.bc" % self.prog.name):
                connection.put(filename, dest + "Programs/Bytecode")
            # inputs
            for filename in glob.glob("Player-Data/Input*-P%d-*" % i):
                connection.put(filename, dest + "Player-Data")
            # key and certificates
            for suffix in ('key', 'pem'):
                connection.put("Player-Data/P%d.%s" % (i, suffix),
                               dest + "Player-Data")
            for filename in glob.glob("Player-Data/*.0"):
                connection.put(filename, dest + "Player-Data")

        import threading
        import random
        threads = []
        for i in range(len(hosts)):
            threads.append(threading.Thread(target=run, args=(i,)))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # execution
        threads = []
        # random port numbers to avoid conflict
        port = 10000 + random.randrange(40000)
        if '@' in hostnames[0]:
            party0 = hostnames[0].split('@')[1]
        else:
            party0 = hostnames[0]
        for i in range(len(connections)):
            run = lambda i: connections[i].run(
                "cd %s; ./%s -p %d %s -h %s -pn %d %s" % \
                (destinations[i], vm, i, self.prog.name, party0, port,
                 ' '.join(args)))
            threads.append(threading.Thread(target=run, args=(i,)))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
