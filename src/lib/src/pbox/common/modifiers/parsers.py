import lief

__all__ = ["parser_handler"]
__parsers__ = ["lief_parser"]


def parser_handler(parser_name):
    def decorator(modifier_func):
        def wrapper(parser=None, executable=None, parsed=None, **kw):
            # parsed is not None only for testing
            if parsed is not None:
                modifier_func(parsed=parsed ,**kw)
                return None
            
            # redefine parser if necessary
            if parser_name not in __parsers__:
                raise ValueError("Parser {parser_name} could not be found")

            # if none was used
            if parser is None:
                parser = globals()[parser_name](executable)

            # if another was used for previous modifiers, close the previous one
            elif parser.__class__.__name__ != parser_name:
                parser.build()
                parser = globals()[parser_name](executable)

            # call the modifier via the parser
            parser(modifier_func, executable=executable, **kw)
            return parser
        return wrapper
    return decorator


class lief_parser():
    name = "lief_parser"
    def __init__(self, executable):
        self.executable = executable
        self.parsed = lief.parse(str(self.executable.destination))
        self.build_instructions = {"imports": False, "dos_stub": False, "patch_imports":False}

    def __call__(self, modifier_func, **kw):
        """Calls the modifier function with the parsed executable

        Args:
            modifier_func (function): Modifier function
        """
        kw.update(parsed=self.parsed)
        out = modifier_func(**kw)
        if out is not None:
            self.build_instructions.update(out)

    def get_sections(self):
        return list(self.parsed.sections)
    
    def compute_checksum(self):
        self.build()
        self.__init__(self.executable)
        return self.parsed.optional_header.computed_checksum

    def build(self):
        builder = lief.PE.Builder(self.parsed)
        builder.build_imports(self.build_instructions["imports"])
        builder.patch_imports(self.build_instructions["patch_imports"])
        builder.build()
        builder.write(str(self.executable.destination))
