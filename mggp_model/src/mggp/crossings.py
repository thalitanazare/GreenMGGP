from abc import ABC, abstractmethod
from deap import gp, base
from mggp.base import Individual
import random


class Crossing(ABC):
    def __init__(self, element):
        self._element = element
        self._toolbox = base.Toolbox()

    @abstractmethod
    def cross(self, ind1, ind2):
        pass

    def _gpConstraint(self, func, *args):
        clone = tuple(map(self._toolbox.clone, args))
        offspring = func(*args)
        for tree in offspring:
            if tree.height > self._element._maxHeight:
                return clone
        return offspring


class CrossLowOnePoint(Crossing):
    """
        Low-level one-point crossover.

        Performs crossover inside a single GP gene (tree). One corresponding gene
        is randomly selected from both parents, and DEAP's gp.cxOnePoint is applied
        to exchange subtrees between these two genes.

        Therefore, this operator modifies the internal structure of one gene while
        keeping the remaining genes unchanged.
    """
    
    def __init__(self, element):
        super().__init__(element)

    def cross(self, ind1: Individual, ind2: Individual) -> tuple[Individual, Individual]:
        if self._element._mode in ['SISO', 'MISO'] or (self._element._mode == 'FIR' and self._element._nOutputs == 1):
            # idx = random.randint(0, len(ind1) - 1)
            idx = random.randint(0, min(len(ind1), len(ind2)) - 1)
            ind1[idx], ind2[idx] = self._gpConstraint(gp.cxOnePoint, ind1[idx], ind2[idx])

            return ind1, ind2
        
        if self._element._mode == 'MIMO' or (self._element._mode == 'FIR' and self._element._nOutputs > 1):
            idx = random.randint(0, min(len(ind1), len(ind2)) - 1)
            idx2 = random.randint(0, min(len(ind1[0]), len(ind2[0])) - 1)
            ind1[idx][idx2], ind2[idx][idx2] = self._gpConstraint(gp.cxOnePoint, ind1[idx][idx2], ind2[idx][idx2])
            
            return ind1, ind2


class CrossLowUniform(Crossing):
    """
        Low-level uniform crossover.

        Performs crossover inside multiple GP genes (trees). Each pair of
        corresponding genes has an independent probability (`indpb`) of undergoing
        DEAP's gp.cxOnePoint crossover.

        Consequently, several genes may have their internal tree structures
        recombined during a single crossover operation.
    """

    def __init__(self, element):
        super().__init__(element)

    def cross(self, ind1: Individual, ind2: Individual) -> tuple[Individual, Individual]:
        if self._element._mode in ['SISO', 'MISO'] or (self._element._mode == 'FIR' and self._element._nOutputs == 1):
            indpb = 0.5
            for i in range(min(len(ind1), len(ind2))):
                if random.random() < indpb:
                    ind1[i], ind2[i] = self._gpConstraint(gp.cxOnePoint, ind1[i], ind2[i])

            return ind1, ind2
        
        if self._element._mode == 'MIMO' or (self._element._mode == 'FIR' and self._element._nOutputs > 1):
            indpb = 0.5
            for o in range(min(len(ind1), len(ind2))):
                for i in range(min(len(ind1[o]), len(ind2[o]))):
                    if random.random() < indpb:
                        ind1[o][i], ind2[o][i] = self._gpConstraint(gp.cxOnePoint, ind1[o][i], ind2[o][i])

            return ind1, ind2


class CrossHighOnePoint(Crossing):
    """
        High-level one-point crossover.

        Performs crossover at the MGGP gene-list level instead of modifying the
        internal structure of GP trees. A crossover position is selected and the
        complete genes after that position are exchanged between the parents.

        Each gene is treated as an indivisible unit: its GP tree is transferred
        without modification.
    """
    def __init__(self, element):
        super().__init__(element)

    def cross(self, ind1: Individual, ind2: Individual) -> tuple[Individual, Individual]:
        if self._element._mode in ['SISO', 'MISO'] or (self._element._mode == 'FIR' and self._element._nOutputs == 1):
            # idx = random.randint(0, len(ind1) - 1)
            idx = random.randint(0, min(len(ind1), len(ind2)) - 1)
            aux = ind1[idx:]
            del ind1[idx:]
            ind1 += ind2[idx:]
            del ind2[idx:]
            ind2 += aux
            return ind1, ind2
        if self._element._mode == 'MIMO' or (self._element._mode == 'FIR' and self._element._nOutputs > 1):
            for o in range(min(len(ind1), len(ind2))):
                idx = random.randint(1, min(len(ind1[o]), len(ind2[o])) - 1)
                aux = ind1[o][idx:]
                del ind1[o][idx:]
                ind1[o] += ind2[o][idx:]
                del ind2[o][idx:]
                ind2[o] += aux
            return ind1, ind2


class CrossHighUniform(Crossing):
    """
        High-level uniform crossover.

        Performs crossover at the MGGP gene-list level. Each pair of corresponding
        genes has an independent probability (`indpb`) of being exchanged between
        the two parents.

        The internal GP tree of each gene remains unchanged; complete genes are
        moved between individuals.
    """

    def __init__(self, element):
        super().__init__(element)

    def cross(self, ind1: Individual, ind2: Individual) -> tuple[Individual, Individual]:
        if self._element._mode in ['SISO', 'MISO'] or (self._element._mode == 'FIR' and self._element._nOutputs == 1):
            indpb = 0.5
            # for i in range(len(ind1)):
            for i in range(min(len(ind1), len(ind2))):
                if random.random() < indpb:
                    aux = ind1[i]
                    ind1[i] = ind2[i]
                    ind2[i] = aux

            return ind1, ind2
        
        if self._element._mode == 'MIMO' or (self._element._mode == 'FIR' and self._element._nOutputs > 1):
            for o in range(min(len(ind1), len(ind2))):
                indpb = 0.5
                for i in range(min(len(ind1[o]), len(ind2[o]))):
                    if random.random() < indpb:
                        aux = ind1[o][i]
                        ind1[o][i] = ind2[o][i]
                        ind2[o][i] = aux

            return ind1, ind2
