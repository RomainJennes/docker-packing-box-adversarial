# -*- coding: UTF-8 -*-
import json
import logging
import re
from argparse import ArgumentParser, RawTextHelpFormatter
from ast import literal_eval
from os.path import abspath, exists, isfile
from pprint import pformat
from shlex import split
from subprocess import call, Popen, PIPE
from sys import argv, exit
from time import perf_counter
from yaml import safe_load


__all__ = ["json", "literal_eval", "pformat", "re", "run", "PACKERS", "PACKERS_FILE"]


DETECTORS      = None
DETECTORS_FILE = "/opt/detectors.yml"
PACKERS        = None
PACKERS_FILE   = "/opt/packers.yml"


def execute(name, **kwargs):
    """ Run an OS command. """
    # from the specific arguments' parsed values and actions, reconstruct the options string
    spec, spec_val = "", kwargs.get('orig_args', {})
    for a in kwargs.get('_orig_args', []):
        n = a.dest
        v = spec_val[n]
        if a.__class__.__name__ == "_StoreTrueAction":
            if v is True:
                spec += " " + a.option_strings[0]
        elif a.__class__.__name__ == "_StoreFalseAction":
            if v is False:
                spec += " " + a.option_strings[0]
        elif a.__class__.__name__ == "_SubParsersAction":
            spec += " " + n
        elif isinstance(v, (list, tuple)):
            for x in v:
                spec += " " + a.option_strings[0] + " " + str(x)
        elif v is not None:
            spec += " " + a.option_strings[0] + " " + str(v)
    cmd = DETECTORS[name].get('command', "/usr/bin/%s {path}" % name.lower())
    exe, opt = cmd.split(" ", 1)
    cmd = (exe + "%s " + opt) % spec
    kwargs['logger'].debug("Command format: " + cmd)
    if kwargs.get('version', False):
        call([cmd.split()[0], "--version"])
        exit(0)
    shell = ">" in cmd
    # prepare the command line and run the tool
    cmd = cmd.format(**kwargs)
    cmd = cmd if shell else split(cmd)
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=shell)
    out, err = proc.communicate()
    return out.decode(), err.decode()


def normalize(*packers):
    """ Normalize the output from a list of values based on the PACKERS list.
    
    :param packers: list of packer-related strings
    """
    if len(packers) == 0 or packers in [(None, ), ("", )]:
        return
    d = {'unknown': -1}
    for s in packers:
        for packer, details in PACKERS.items():
            for p in ["(?i)" + packer] + details.get('aliases', []):
                if re.search(p, s):
                    p = packer.lower()
                    d.setdefault(p, 0)
                    d[p] += 1
                    break
    m = [k for k, v in d.items() if v == max(d.values())]
    return m[0] if len(m) == 1 else "unknown"  # cannot decide when multiple maxima


def run(name, exec_func=execute, parse_func=lambda x, **kw: x, stderr_func=lambda x, **kw: x, parser_args=[],
        normalize_output=True, binary_only=False, weak_assumptions=False, **kwargs):
    """ Run a tool and parse its output.
    
    It also allows to parse stderr and to normalize the output.
    
    :param name:             name of the tool
    :param exec_func:        function for executing the tool
    :param parse_func:       function for parsing the output of stdout
    :param stderr_func:      function for handling the output of stderr
    :param parser_args:      additional arguments for the parser ; format: [(args, kwargs), ...]
    :param normalize_output: normalize the final output based on a base of items
    :param binary_only:      specify that the tool only handles binary classes (i.e. no packer name)
    :param weak_assumptions: specify that the tool has options depending on weak assumptions (e.g. suspicions)
    
    The parse_func shall take the output of stdout and return either a parsed value or None (if no relevant result).
    The stderr_func shall take the output of stderr and return either a parsed error message or None (if no error).
    """
    global DETECTORS, DETECTORS_FILE, PACKERS, PACKERS_FILE
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter, add_help=False)
    opt = parser.add_argument_group("decision arguments")
    exe_type = kwargs.pop('exe_type', "exe")
    parser.add_argument(exe_type, help=kwargs.pop('exe_help', "path to executable"))
    # handle binary-only tools
    if binary_only:
        normalize_output = False
        try:
            argv.remove("--binary")
        except ValueError:
            pass
    else:
        opt.add_argument("--binary", action="store_true", help="output yes/no instead of packer's name")
    # handle weak assumption, e.g. when a tool can output detections and suspicions ; when setting --weak, it will also
    #  consider suspicions for the final decision
    if weak_assumptions:
        opt.add_argument("--weak", action="store_true", help="use weak assumptions for processing")
    else:
        try:
            argv.remove("--weak")
        except ValueError:
            pass
    # handle other specific options
    spec, spec_opt = parser.add_argument_group("original arguments"), []
    for args, kw in parser_args:
        spec_opt.append(spec.add_argument(*args, **kw))
    # handle help options
    if "--version" in argv:
        argv[1:] = ["DOESNOTEXIST", "--version"]
    extra = parser.add_argument_group("extra arguments")
    extra.add_argument("-b", "--benchmark", action="store_true", help="enable benchmarking")
    extra.add_argument("-h", "--help", action="help", help="show this help message and exit")
    extra.add_argument("-v", "--verbose", action="store_true", help="display debug information")
    extra.add_argument("--version", action="store_true", help="show version and exit")
    extra.add_argument("--detectors-file", default=DETECTORS_FILE, help="path to detectors YAML")
    if normalize_output:  # the PACKERS list is only required when normalizing
        extra.add_argument("--packers-file", default=PACKERS_FILE, help="path to packers YAML")
    a = parser.parse_args()
    # put parsed values and specific arguments' actions in dedicated namespace variables
    a.orig_args, a._orig_args = {}, spec_opt
    for opt in spec_opt:
        n = opt.dest
        a.orig_args[n] = getattr(a, n)
        delattr(a, n)
    if binary_only:
        a.binary = True
    logging.basicConfig(format="[%(levelname)s] %(message)s")
    a.logger = logging.getLogger(name.lower())
    a.logger.setLevel([logging.INFO, logging.DEBUG][a.verbose])
    p = a.path = abspath(getattr(a, exe_type))
    if getattr(a, exe_type) != "DOESNOTEXIST" and not exists(p):
        print("[ERROR] file not found")
        exit(1)
    # load related dictionaries
    DETECTORS_FILE = a.detectors_file
    with open(DETECTORS_FILE) as f:
        DETECTORS = safe_load(f.read())
    if normalize_output:
        PACKERS_FILE = a.packers_file
        with open(PACKERS_FILE) as f:
            PACKERS = safe_load(f.read())
    # handle version display
    if a.version:
        v = DETECTORS[name].get('version')
        if v:
            if isfile(v):
                with open(v) as f:
                    print(f.read().strip())
            else:
                print(v)
            exit(0)
        exec_func(name, version=True, data=DETECTORS[name])
    # execute the tool
    t1 = perf_counter()
    out, err = exec_func(name, **vars(a))
    dt = perf_counter() - t1
    # now handle the result if no error
    err = stderr_func(err, **vars(a))
    if err:
        a.logger.error(err)
        exit(1)
    else:
        p = parse_func(out, **vars(a))
        if a.verbose and len(out) > 0:
            a.logger.debug("Output:\n" + ("\n".join(out) if isinstance(out, (list, tuple)) else \
                                          json.dumps(out, indent=4) if isinstance(out, dict) else str(out)) + "\n")
        if normalize_output:
            if not isinstance(p, list):
                p = [p]
            p = normalize(*p)
            if a.binary:
                p = str(p is not None)
        if p is not None:
            if a.benchmark:
                p += " " + str(dt)
            print(p)


def version(string):
    def _wrapper(f):
        def _subwrapper(*args, **kwargs):
            if kwargs.get('version', False):
                print(string)
                exit(0)
            return f(*args, **kwargs)
        return _subwrapper
    return _wrapper

