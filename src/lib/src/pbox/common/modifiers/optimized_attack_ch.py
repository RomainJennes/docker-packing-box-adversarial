from ...learning.model import open_model
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack
#from .parsers import parser_handler
from torch import Tensor
from tinyscript.helpers import Path
import numpy as np

def optimized_attack(modifier_names, grid, model, **model_kw):
    from ..modifiers import Modifiers
    
    space_size = sum([len(ps) for ps in grid.values()]) + len(modifier_names)
        
    def _optim(executable, namespace={}, **modifier_kw):
        modifiers = [Modifiers.registry[executable.format][m] for m in modifier_names]
        model_fn = ClassiferWrapper(model, executable, modifiers, grid, namespace, modifier_kw, model_kw)
        t_adv = hop_skip_jump_attack(model_fn,
                                      Tensor([[0]*space_size]),
                                      2,
                                      contraint=2)
        model_fn.cleanup()
        model_fn.apply_alterations(t_adv, executable)
        
    return _optim

class ClassiferWrapper:
    
    def __init__(self, model_name, executable, modifiers, grid, namespace, modifier_kw, model_kw):
        self.model = open_model(model_name)
        self.executable = executable
        self.original_exe = executable.destination
        self.modifiers = modifiers
        self.grid = {m.name: [] for m in modifiers}
        self.grid.update(grid)
        self.namespace = namespace
        self.modifier_kw = modifier_kw
        self.model_kw = model_kw
    
    def __call__(self, t):
        pred = []
        print(t)
        for i in range(t.shape[0]):
            self.executable._destination = Path(str(self.original_exe) + str(i))
            self.executable.copy(overwrite=True)
            self.executable.destination.chmod(0o600)
            self.apply_alterations(t[i], self.executable)
            pred.append(self.model.predict(str(self.executable.destination), **self.model_kw))
        print(pred)
        return Tensor([pred]).T
    
    def cleanup(self):
        self.executable.destination.remove()
        self.executable._destination = self.original_exe
        #self.executable.copy()
        self.executable.chmod(0o600)
    
    def apply_alterations(self, t, exe):
        i = 0
        parser = None
        for modifier in self.modifiers:
            namespace = self.namespace.copy()
            for param in self.grid[modifier.name]:
                n_vals = len(self.grid[modifier.name][param])
                namespace[param] = self.grid[modifier.name][param][min(max(int((t[i]*n_vals).floor()), 0), n_vals-1)]
                print(f"{param}: {namespace[param]}")
                i += 1
            print(f"Applying {modifier.name}: {t[i] > 0.5}")
            if t[i] > 0.5:
                parser = modifier(namespace, parser=parser, namespace=namespace, executable=exe)
            i+=1
        if parser is not None:
            parser.build()
