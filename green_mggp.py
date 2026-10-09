"""Green MGGP: bi-objective MGGP with a platform-weighted arithmetic cost.

Built on top of the ``mggp`` package (R. Avila, github.com/RafaelGAT108/mggp_model)
without modifying it. Two additions:

1. ``arithmetic_cost`` -- the arithmetic cost of the polynomial model that is actually
   executed, in addition-equivalents,  C_rho = a + rho * b,  with a = p-1 additions,
   b = sum_i d_i multiplications and rho the cost of a multiplication relative to an
   addition on the target platform. ``RHO_OPS = 1`` counts operations; ``RHO_KARATSUBA
   = B**(alpha-1) = 729/64`` is the bit-level Green-Box weighting (Karatsuba, B = 64),
   whose cost equals the original Green-Box complexity  B a + B**alpha b  divided by B.
2. ``GreenMGGP`` -- Pareto-based (NSGA-II) parent selection and survival, so that
   the cost objective drives the search instead of only breaking ties.
"""
from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np
from deap import gp
from deap.tools import selNSGA2, sortNondominated
from deap.tools.emo import assignCrowdingDist
from mggp import MGGP

B_BITS = 64                 # operand length (binary64)
ALPHA = math.log2(3)        # Karatsuba multiplication exponent, ~1.585
RHO_OPS = 1.0               # multiplication costs one addition: operation count
RHO_KARATSUBA = 729 / 64    # B**(alpha-1) for B = 64, exact in binary (= 3**6 / 2**6)
COST_SCENARIOS = {"ops": RHO_OPS, "karatsuba": RHO_KARATSUBA}


# ---------------------------------------------------------------------------
# Green-Box arithmetic cost
# ---------------------------------------------------------------------------
def gene_monomial(tree: gp.PrimitiveTree) -> tuple:
    """Canonical monomial of one gene: sorted tuple of ((variable, shift), power).

    ``mul`` multiplies factors, ``q<j>`` delays the whole sub-tree by j samples.
    The shift is relative to the terminal's base lag in the package
    (y1 -> y[k-1], u1 -> u[k]), which is sufficient to decide equality.
    """
    def parse(i: int, shift: int):
        node = tree[i]
        if isinstance(node, gp.Terminal):
            return Counter({(str(node.value), shift): 1}), i + 1
        name = node.name
        if name.startswith("q") and node.arity == 1:
            return parse(i + 1, shift + int(name[1:]))
        if name == "mul":
            left, j = parse(i + 1, shift)
            right, k = parse(j, shift)
            return left + right, k
        raise ValueError(f"arithmetic_cost only supports 'mul' and backshift operators, got '{name}'")

    factors, _ = parse(0, 0)
    return tuple(sorted(factors.items()))


def canonical_terms(individual) -> frozenset:
    """Set of distinct monomials (duplicated genes collapse into one regressor)."""
    return frozenset(gene_monomial(g) for g in individual)


def model_size(individual) -> tuple[int, int]:
    """(number of distinct non-constant regressors, total degree sum_i d_i)."""
    terms = canonical_terms(individual)
    return len(terms), sum(sum(p for _, p in t) for t in terms)


def arithmetic_cost(individual, rho: float = RHO_OPS) -> float:
    """Arithmetic cost of the executed NARX polynomial in addition-equivalents.

    The model is  y = theta_0 + sum_{i=1}^{p-1} theta_i * phi_i,  phi_i a monomial of
    degree d_i. Direct evaluation needs a = p-1 additions and b = sum_i d_i
    multiplications (coefficient times d_i factors); a multiplication costs rho additions.
    """
    n_terms, total_degree = model_size(individual)
    return n_terms + rho * total_degree


def make_cost(rho: float):
    """Cost function with a fixed multiplication weight, for ``new_evaluation``."""
    def cost(individual) -> float:
        return arithmetic_cost(individual, rho)
    cost.rho = rho
    cost.__name__ = f"arithmetic_cost_rho_{rho:g}"
    return cost


def green_cost(individual, B: int = B_BITS, alpha: float = ALPHA) -> float:
    """Original Green-Box complexity  B a + B**alpha b  (bit operations); equals
    B * arithmetic_cost(individual, RHO_KARATSUBA). Kept for reference."""
    n_terms, total_degree = model_size(individual)
    return n_terms * B + (B ** alpha) * total_degree


def count_times(individual, times_weight: float = 0.66) -> float:
    """Complexity used in the package tutorial (kept only for comparison)."""
    return sum(str(g).count("mul") * times_weight for g in individual)


# ---------------------------------------------------------------------------
# Green MGGP: Pareto-based selection on (error, green cost)
# ---------------------------------------------------------------------------
def _finite(ind) -> bool:
    return ind.fitness.valid and all(np.isfinite(ind.fitness.values))


class GreenMGGP(MGGP):
    """MGGP with the arithmetic cost (weight ``rho``) as second objective and NSGA-II survival.

    Representation, genetic operators, least-squares estimation and error
    predictors are those of the ``mggp`` package. Only selection changes:
    parents by crowded binary tournament, survivors by elitist (mu+lambda)
    NSGA-II on the union of parents and offspring, after removing structural
    duplicates (same canonical set of monomials).
    """

    def __init__(self, *args, rho: float = RHO_OPS, **kwargs):
        kwargs.setdefault("new_evaluation", make_cost(rho))
        super().__init__(*args, **kwargs)

    def evaluation(self, ind):
        fit = super().evaluation(ind)
        return fit if np.all(np.isfinite(fit)) else (np.inf, np.inf)

    @staticmethod
    def _assign_rank_crowding(pop):
        for rank, front in enumerate(sortNondominated(pop, len(pop))):
            assignCrowdingDist(front)
            for ind in front:
                ind.rank = rank

    @staticmethod
    def _crowded_tournament(pop, n):
        def better(a, b):
            if a.rank != b.rank:
                return a if a.rank < b.rank else b
            return a if a.fitness.crowding_dist >= b.fitness.crowding_dist else b
        return [better(*random.sample(pop, 2)) for _ in range(n)]

    def _survive(self, pool, n):
        valid = [ind for ind in pool if _finite(ind)]
        unique, repeated, seen = [], [], set()
        for ind in valid:
            key = canonical_terms(ind)
            (repeated if key in seen else unique).append(ind)
            seen.add(key)
        # diagnostic only (no effect on the search): size of the first front of the
        # de-duplicated union, the quantity |U_1| in the non-deterioration theorem
        if unique:
            self.front1_sizes = getattr(self, "front1_sizes", [])
            self.front1_sizes.append(len(sortNondominated(unique, len(unique), first_front_only=True)[0]))
        if len(unique) >= n:
            return selNSGA2(unique, n)
        # not enough distinct structures: keep all of them, fill with duplicates,
        # then (only if still short) with non-finite individuals
        need = n - len(unique)
        fill = selNSGA2(repeated, min(need, len(repeated))) if repeated else []
        invalid = [ind for ind in pool if not _finite(ind)]
        return unique + fill + invalid[: need - len(fill)]

    def step(self, gen_number: int) -> None:
        from copy import deepcopy

        pop = [ind for ind in self._pop if _finite(ind)] or self._pop
        self._assign_rank_crowding(pop)
        offspring = [deepcopy(ind) for ind in self._crowded_tournament(pop, self.populationSize)]

        for i in range(0, len(offspring) - 1, 2):
            if np.random.random() < self.crossoverRate:
                cross = random.choice(self._crossList)
                offspring[i], offspring[i + 1] = cross.cross(offspring[i], offspring[i + 1])
                self._delAttr(offspring[i])
                self._delAttr(offspring[i + 1])
        for i in range(len(offspring)):
            if np.random.random() < self.mutationRate:
                mut = random.choice(self._mutList)
                offspring[i], = mut.mutate(offspring[i])
                self._delAttr(offspring[i])

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(self._toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        self._pop = self._survive(self._pop + offspring, self.populationSize)
        self._hof.update(self._pop)
        self._logbook.record(gen=gen_number + 1, evals=len(invalid),
                             fitness=self._stats.compile(self._pop))
