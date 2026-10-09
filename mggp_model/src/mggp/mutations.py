from abc import ABC, abstractmethod
from deap import gp, base
from mggp.base import Individual
import random


class Mutation(ABC):
    def __init__(self, element):
        self._element = element
        self._toolbox = base.Toolbox()

    @abstractmethod
    def mutate(self, ind):
        pass

    def _gpConstraint(self, func, tree):
        clone = self._toolbox.clone(tree)
        offspring = func(tree)[0]
        if offspring.height > self._element._maxHeight:
            return clone,
        return offspring,

    
    # def generate_terminal_only(self, pset: gp.PrimitiveSet, type_=None) -> gp.PrimitiveTree:
    #         """Retorna uma árvore com apenas um nó terminal escolhido aleatoriamente"""
    #
    #         terminal_index = random.randint(1, self._element._nVar)
    #
    #         if self._element._mode == "FIR":
    #             terminal_name = f"u{terminal_index}"
    #
    #         else:
    #
    #             if terminal_index <= (self._element._nVar - self._element._nOutputs):
    #                 terminal_name = f"u{terminal_index}"
    #
    #             else:
    #                 terminal_name = f"y{terminal_index}"
    #
    #         terminal_node = pset.mapping[terminal_name]
    #         return gp.PrimitiveTree([terminal_node])
        
    
    def genGrowLimitedNodes(self, pset: gp.PrimitiveSet, min_depth: int, max_depth: int, max_nodes: int, type_=None) -> list:

        while True:
            expr = gp.genHalfAndHalf(pset, min_depth, max_depth, type_=type_)
            tree = gp.PrimitiveTree(expr)
            if len(tree) <= max_nodes:
                return expr


class MutGPOneTree(Mutation):
    """
        Single-tree GP mutation.

        Applies a GP subtree mutation to one randomly selected gene (tree) of the
        individual. The selected tree is not completely replaced; instead, DEAP's
        gp.mutUniform selects a random node/subtree inside the tree and replaces
        that subtree with a newly generated expression.

        The newly generated subtree is created using a constrained generator,
        limiting both tree depth and number of nodes.
    """

    def __init__(self, element):
        super().__init__(element)
        # self._toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=self._element._maxHeight - 1)
        
        # self._toolbox.register("expr_mut", self.generate_terminal_only)
        self._toolbox.register("expr_mut", self.genGrowLimitedNodes, min_depth=0, max_depth=self._element._maxHeight-1, max_nodes=20)
        self._toolbox.register("mutateGP", gp.mutUniform, expr=self._toolbox.expr_mut, pset=self._element.pset)

    def mutate(self, ind: Individual) -> Individual:
        if self._element._mode in ['SISO', 'MISO']:
            idx = random.randint(0, len(ind) - 1)
            ind[idx] = self._gpConstraint(self._toolbox.mutateGP, ind[idx])[0]

        if self._element._mode == 'MIMO' or (self._element._mode == 'FIR' and self._element._nOutputs > 1):
            for o in range(len(ind)):
                idx = random.randint(0, len(ind[o]) - 1)
                ind[o][idx] = self._gpConstraint(self._toolbox.mutateGP, ind[o][idx])[0]
        return ind,


class MutGPUniform(Mutation):
    """
        Uniform GP mutation.

        Applies GP subtree mutation independently to each gene with probability
        `indpb`. Every selected gene undergoes DEAP's gp.mutUniform mutation, which
        replaces a randomly selected subtree with a newly generated expression.

        Unlike MutGPOneTree, the number of mutated genes is not fixed. Depending on
        the random draws, no gene, one gene, or several genes may be modified in a
        single mutation operation.
    """

    def __init__(self, element):
        super().__init__(element)
        self._toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=self._element._maxHeight - 1)
        # self._toolbox.register("expr_mut", self.generate_terminal_only)
        # self._toolbox.register("expr_mut", self.genGrowLimitedNodes, min_depth=0, max_depth=self._element._maxHeight-1, max_nodes=20)
        self._toolbox.register("mutateGP", gp.mutUniform, expr=self._toolbox.expr_mut, pset=self._element.pset)

    def mutate(self, ind: Individual) -> Individual:
        if self._element._mode in ['SISO', 'MISO']:
            indpb = 0.50
            for i in range(len(ind)):
                if random.random() < indpb:
                    ind[i] = self._gpConstraint(self._toolbox.mutateGP, ind[i])[0]
                    
        if self._element._mode == 'MIMO' or (self._element._mode == 'FIR' and self._element._nOutputs > 1):
            indpb = 0.50
            for o in range(len(ind)):
                for i in range(len(ind[o])):
                    if random.random() < indpb:
                        ind[o][i] = self._gpConstraint(self._toolbox.mutateGP, ind[o][i])[0]
        return ind,


class MutGPReplace(Mutation):
    """
        Complete-gene replacement mutation.

        Replaces an entire GP gene (tree) with a newly generated random program.
        Unlike subtree-based mutations, no structural information from the selected
        gene is preserved.

        This is therefore a stronger and more disruptive mutation operator than
        MutGPOneTree or MutGPUniform.
    """
    def __init__(self, element):
        super().__init__(element)

    def mutate(self, ind: Individual) -> Individual:
        if self._element._mode in ['SISO', 'MISO']:

            idx = random.randint(0, len(ind) - 1)
            ind[idx] = self._element._toolbox._program()


        if self._element._mode == 'MIMO' or (self._element._mode == 'FIR' and self._element._nOutputs > 1):
            for o in range(len(ind)):
                idx = random.randint(0, len(ind[o]) - 1)
                ind[o][idx] = self._element._toolbox._program()
        return ind,
