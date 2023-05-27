from ...learning.model import open_model
import cleverhans.torch.attacks as ch_attacks
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.sparse_l1_descent import sparse_l1_descent
from cleverhans.torch.attacks.spsa import spsa
from tinyscript import logging
from random import random

#from .parsers import parser_handler
from torch import Tensor
from tinyscript.helpers import Path
import numpy as np


def optimized_attack(modifier_names, grid, model, attack="FGM", print_results=False, **model_kw):
    from ..modifiers import Modifiers
    
    for m in grid:
        for p in grid[m]:
            grid[m][p] = list(grid[m][p])
        
    space_size = sum([len(ps) for ps in grid.values()]) + len(modifier_names)
        
    def _optim(executable, namespace={}, **modifier_kw):
        modifiers = [Modifiers.registry[executable.format][m] for m in modifier_names]
        model_fn = ClassiferWrapper(model, executable, modifiers, grid, namespace, modifier_kw, model_kw, print_results=print_results)
        
        if attack == "HopSkipJump":
            t_adv = hop_skip_jump_attack(model_fn,
                                        Tensor([[0]*space_size]),
                                        2,
                                        constraint=2,
                                        gamma=2,
                                        num_iterations=10,
                                        initial_num_evals=10,
                                        max_num_evals=100)
        elif attack == "FGM":
            t_adv = fast_gradient_method(model_fn,
                                        Tensor([[0]*space_size]),
                                        eps=0.25,
                                        norm=2)
        elif attack == "SPSA":
            t_adv = spsa(model_fn,
                        Tensor([[random() for _ in range(space_size)]]),
                        eps=1,
                        norm=2,
                        nb_iter=200,
                        spsa_samples=32,
                        learning_rate=0.01,
                        delta=0.1,
                        clip_min=0,
                        clip_max=1,
                        is_debug=True
                        )
        
        model_fn.cleanup()
        model_fn.apply_alterations(t_adv[0], executable, verbose=True)
        
    return _optim

class ClassiferWrapper:
    
    def __init__(self, model_name, executable, modifiers, grid, namespace, modifier_kw, model_kw, print_results=False):
        self.model = open_model(model_name)
        self.executable = executable
        self.original_exe = executable.destination
        self.modifiers = modifiers
        self.grid = {m.name: [] for m in modifiers}
        self.grid.update(grid)
        self.namespace = namespace
        self.modifier_kw = modifier_kw
        self.model_kw = model_kw
        self.print_results = print_results
    
    def __call__(self, t):

        pred = np.zeros((t.shape[0],2))
        if self.print_results:
            print(t)
        for i in range(t.shape[0]):
            self.model.logger.setLevel(30)
            self.executable._destination = Path(str(self.original_exe) + str(i))
            self.executable.copy(overwrite=True)
            self.executable.destination.chmod(0o600)
            self.apply_alterations(t[i], self.executable)
            pred[i][1] = self.model.predict(str(self.executable.destination), **self.model_kw)
            pred[i][0] = 1 - pred[i][1]
            self.executable.destination.remove()
        if self.print_results:
            print(Tensor(pred))
        return Tensor(pred)
    
    def cleanup(self):
        #self.executable.destination.remove()
        self.executable._destination = self.original_exe
        #self.executable.copy()
        self.executable.chmod(0o600)
    
    def apply_alterations(self, t, exe, verbose=False):
        i = 0
        parser = None
        for modifier in self.modifiers:
            #namespace = self.namespace.copy()
            for param in self.grid[modifier.name]:
                n_vals = len(self.grid[modifier.name][param])
                modifier.parameters[param] = self.grid[modifier.name][param][min(max(int((t[i]*n_vals).floor()), 0), n_vals-1)]

                i += 1
            if t[i] > 0.5:
                if verbose:
                    print("%s:" % modifier.name)
                    print(modifier.parameters)
                try:
                    parser = modifier(self.namespace, parser=parser, namespace=self.namespace, executable=exe)
                except Exception as e:
                    self.logger.warning("%s: %s" % ("optimized_attack", str(e)))
            i+=1
        if parser is not None:
            parser.build()
