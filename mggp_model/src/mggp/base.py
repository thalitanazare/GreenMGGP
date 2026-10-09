from __future__ import annotations
from numpy.typing import NDArray
import pickle
from functools import partial
from abc import abstractmethod
from typing import List
import sys
import random
from deap import gp, creator, base, tools
import operator
import numpy as np
import re
from .predictors import miso_OSA, miso_FreeRun, miso_MShooting, mimo_CLASSIFY, miso_FIR_INSTANT, mimo_FIR_INSTANT, miso_CLASSIFY
from .predictors import mimo_OSA, mimo_FreeRun, mimo_MShooting, mimo_FIR_MShooting, mimo_FIR_FreeRun
import warnings
warnings.filterwarnings('ignore')

def _roll(*args, i):
    return np.roll(*args, shift=i)

def sign(X1, X2):
    return np.sign(X1-X2)

def tanh(X1, X2):
    return np.tanh(X1-X2)

def sin(X1):
    return np.sin(X1)

def div(X1, X2):
    return X1 / X2

def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error (MSE)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    return np.mean((y_true - y_pred) ** 2)


def accuracy_score(y_true, y_pred):
    """
    Accuracy Score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    return np.mean(y_true == y_pred)


def log_loss(y_true, y_pred, eps=1e-15):
    """
    Log Loss (Cross-Entropy)

    Suporta:
    - Binário: y_true em {0,1}, y_pred probabilidade da classe 1
    - Multiclasse: y_true como labels inteiros ou one-hot, y_pred como probs
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Clipping para evitar log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Caso binário
    if y_pred.ndim == 1:
        if y_true.shape != y_pred.shape:
            raise ValueError("Shape mismatch for binary case")

        return -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

    # Caso multiclasse
    elif y_pred.ndim == 2:
        n_samples, n_classes = y_pred.shape

        # Se y_true está como labels (inteiros)
        if y_true.ndim == 1:
            if len(y_true) != n_samples:
                raise ValueError("y_true length mismatch")

            return -np.mean(np.log(y_pred[np.arange(n_samples), y_true]))

        # Se y_true é one-hot
        elif y_true.ndim == 2:
            if y_true.shape != y_pred.shape:
                raise ValueError("Shape mismatch for one-hot case")

            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

        else:
            raise ValueError("Invalid y_true format")

    else:
        raise ValueError("Invalid y_pred dimensions")



class Individual(list):
    def __init__(self, data: List = []):
        super().__init__(data)
        self.funcs = []
        self.lagMax = None
        self.theta = np.array([])
        self.terminals = ''
        self.nTerms = 0
        self.logistic_model = None 

    @abstractmethod
    def makeRegressors(self, y: NDArray, u: NDArray):
        pass

    @abstractmethod
    def leastSquares(self, y: NDArray, u: NDArray):
        pass


    def constrained_least_squares(self, y: NDArray, p: NDArray, constraints=None) -> NDArray:

        yd = y[self.lagMax:]
        
        theta_ls = np.linalg.lstsq(p, yd, rcond=None)[0]
        
        if constraints is None:
            self.theta = theta_ls
            return self.theta
        
        S = constraints['S']  # Matriz de restrições
        c = constraints['c'].reshape(-1, 1)  # Vetor de restrições
        
        pT_p = p.T @ p
        pT_p_inv = np.linalg.pinv(pT_p)

        term1 = pT_p_inv @ S.T
        term2 = np.linalg.pinv(S @ pT_p_inv @ S.T)
        term3 = S @ theta_ls - c

        theta_cls = theta_ls - term1 @ term2 @ term3

        # R = constraints.get("R", np.zeros((p.shape[1], p.shape[1])))
        # lambda_ridge = constraints.get("lambda", 0.0)
        #
        # A = p.T @ p + lambda_ridge * R
        # A_inv = np.linalg.pinv(A)
        #
        # theta_ridge = A_inv @ (p.T @ yd)
        #
        # term1 = A_inv @ S.T
        # term2 = np.linalg.pinv(S @ A_inv @ S.T)
        # term3 = S @ theta_ridge - c
        #
        # theta_cls = theta_ridge - term1 @ term2 @ term3

        self.theta = theta_cls
        return self.theta
    

    def identify_term_clusters(self, y: NDArray, u: NDArray) -> dict:
        """Identify regressor term clusters used by the hysteretic constraints (Property 1).

        The paper groups parameters into clusters and enforces:
        - sum of *linear output* parameters = 1
        - sum of all other (non-ϕ) clusters = 0
        - terms containing ϕ-functions are ignored in steady-state constraints

        Important: `makeRegressors` in this codebase includes a leading bias column of ones.
        The paper's model form (Eq. 10) has no explicit intercept. If we keep a bias term,
        to preserve a *continuum of equilibria* we must constrain its parameter to 0.
        """
        p = self.makeRegressors(y, u)
        if not isinstance(p, np.ndarray):
            raise TypeError(
                "identify_term_clusters currently supports scalar-output regressors matrices (SISO/MISO/FIR). "
                "For MIMO, run this per-output (one Ψ per output) or refactor makeRegressors accordingly."
            )

        n_terms = p.shape[1]

        clusters = {
            "bias": [],
            "linear_output": [],
            "linear_input": [],
            "nonlinear_y": [],
            "nonlinear_u": [],
            "cross_terms": [],
            "phi_terms": [],
        }

        for term_index in range(n_terms):
            clusters[self._classify_term(term_index)].append(term_index)

        return clusters


    @staticmethod
    def _node_is_q(node) -> bool:
        return isinstance(node, gp.Primitive) and re.fullmatch(r"q\d+", getattr(node, "name", "")) is not None


    @staticmethod
    def _node_is_phi(node) -> bool:
        return isinstance(node, gp.Primitive) and getattr(node, "name", "") in ("subtraction", "sign")


    @staticmethod
    def _terminal_value_str(term) -> str:
        v = getattr(term, "value", None)
        if v is None:
            v = getattr(term, "name", "")
        return str(v)


    def _tree_contains_var(self, tree, var_prefix: str) -> bool:
        for node in tree:
            if isinstance(node, gp.Terminal) and self._terminal_value_str(node).startswith(var_prefix):
                return True
        return False


    def _tree_has_only_q_and_var(self, tree: gp.PrimitiveTree, var_prefix: str, require_at_least_one_q: bool = True) -> bool:
        q_count = 0
        term_count = 0

        for node in tree:
            if isinstance(node, gp.Primitive):
                if self._node_is_q(node):
                    q_count += 1
                else:
                    return False
            elif isinstance(node, gp.Terminal):
                term_count += 1
                if not self._terminal_value_str(node).startswith(var_prefix):
                    return False
            else:
                return False

        if term_count != 1:
            return False
        if require_at_least_one_q and q_count == 0:
            return False
        return True


    def _is_linear_term(self, tree_str: str = None, tree=None) -> bool:
        """Backward-compatible linearity check.

        A term is considered linear (for clustering) if it is a pure delay chain of one variable:
        q*(y) or q*(u). Any other primitive makes it nonlinear.
        """
        if tree is None:
            return False
        
        return (
            self._tree_has_only_q_and_var(tree, "y", require_at_least_one_q=True)
            or self._tree_has_only_q_and_var(tree, "u", require_at_least_one_q=True)
        )

    def _classify_term(self, term_index: int) -> str:
        """
        Classifica um termo baseado em sua estrutura
        """
        if term_index == 0:
            # return 'linear_output'
            return 'bias'
        
        # Para MISO/FIR
        if hasattr(self, 'funcs') and len(self.funcs) > term_index - 1:
            tree_str = str(self[term_index - 1]).lower()
            
            # Verifica se contém funções phi
            if 'subtraction' in tree_str or 'sign' in tree_str:
                return 'phi_terms'
            
            if 'y' in tree_str and 'u' not in tree_str:
                if self._is_linear_term(tree_str):
                    return 'linear_output'
                else:
                    return 'nonlinear_y'
                    
            elif 'u' in tree_str and 'y' not in tree_str:
                if self._is_linear_term(tree_str):
                    return 'linear_input' 
                else:
                    return 'nonlinear_u'
                    
            elif 'y' in tree_str and 'u' in tree_str:
                return 'cross_terms'
        
        return 'phi_terms'  # Default


    def _is_linear_term(self, tree_str: str) -> bool:
        """Verifica se o termo é linear (apenas multiplicações básicas)"""
        non_linear_ops = ['sign', 'subtraction', 'mul(mul', 'mul(q', 'mul(']
        return not any(op in tree_str for op in non_linear_ops)
    

    def hysteretic_constrained_ls(self, y: NDArray, u: NDArray) -> NDArray:
        """Constrained LS enforcing the hysteretic equilibrium conditions (Property 1).

        Enforced constraints (paper Section 3):
        - sum(theta_linear_output) = 1
        - sum(theta_cluster) = 0 for all other non-ϕ clusters

        Additionally (because this codebase includes a bias term):
        - theta_bias = 0
        """
        clusters = self.identify_term_clusters(y, u)

        constraints = []
        p = self.makeRegressors(y, u)
        length_model = p.shape[1]

        if clusters["bias"]:
            s = np.zeros(length_model)
            s[clusters["bias"]] = 1.0
            constraints.append((s, 0.0))

        if not clusters["linear_output"]:
            raise ValueError("No linear output term (pure delay of y) found. Property-1 constraints cannot be enforced.")

        s = np.zeros(length_model)
        s[clusters["linear_output"]] = 1.0
        constraints.append((s, 1.0))

        for cluster_name in ("linear_input", "cross_terms", "nonlinear_y", "nonlinear_u"):
            idxs = clusters[cluster_name]
            if not idxs:
                continue
            s = np.zeros(length_model)
            s[idxs] = 1.0
            constraints.append((s, 0.0))

        # for cluster_name in ("linear_input",):
        #     idxs = clusters[cluster_name]
        #     if not idxs:
        #         continue
        #     s = np.zeros(length_model)
        #     s[idxs] = 1.0
        #     constraints.append((s, 1.0))

        S = np.vstack([c[0] for c in constraints])
        c = np.array([c[1] for c in constraints])

        lambda_phi = 0

        R = np.zeros((length_model, length_model))

        phi_idxs = clusters["phi_terms"]
        R[phi_idxs, phi_idxs] = 1.0

        constraints = {
            "S": S,
            "c": c,
            # "R": R,
            # "lambda": lambda_phi
        }

        return self.constrained_least_squares(y, p, constraints)


    def predict_proba(self, mode: str ="INSTANT", *args: tuple) -> tuple[NDArray, NDArray]:
        """Predição de probabilidades para classificação"""

        def one_hot_argmax(x: NDArray) -> NDArray:

            result = np.zeros_like(x)
            result[np.argmax(x)] = 1
            
            return result
    
        def softmax(x: NDArray) -> NDArray:
            """Calcula softmax para cada linha do vetor de entrada x."""

            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
        
        if mode == "INSTANT":
            X_regressors, y_true = mimo_CLASSIFY(self, *args)
            # y_true = y_true[:len(y_true)-1] # para a classificação FIR precisa desse termo. Já para a MIMO não.
        # elif mode == "FreeRun":
        #     X_regressors, y_true = mimo_FreeRun(self, *args)
        
        elif mode == "MShooting":
            X_regressors, y_true = mimo_MShooting(self, *args)
        
        else:
            raise Exception("Choose a mode between: INSTANT")
        
        if self.logistic_model is not None:            
            # probabilities = np.array([one_hot_argmax(x) for x in X_regressors])
            pred_one_hot_encode = np.array([one_hot_argmax(softmax(x)) for x in X_regressors])
            # pred_labels = np.array([np.argmax(prob) for prob in pred_one_hot_encode])
            return pred_one_hot_encode, y_true
        
        else:
            raise Exception("Logistic regression model not trained!")


    def predict_classes(self, mode: str ="INSTANT", *args: tuple) -> tuple[NDArray, NDArray]:
        """Predição de classes"""
        
        # probabilities, y_true = self.predict_proba(mode, *args)
        # predicted_classes = np.argmax(probabilities, axis=1)

        predicted_classes, y_true = self.predict_proba(mode, *args)
        
        return predicted_classes, y_true


    def score_classification(self, yd: NDArray, yp: NDArray, mode: str ="accuracy") -> float:
        """Métricas de avaliação para classificação"""
        
        if mode == "accuracy":
            return accuracy_score(yd, yp)
        
        elif mode == "log_loss":
            # Para log_loss, precisamos das probabilidades
            return log_loss(yd, yp)
        
        elif mode == "f1_macro":
            from sklearn.metrics import f1_score
            return f1_score(yd, yp, average='macro')
        
        else:
            raise ValueError("Choose a valid metric: accuracy, log_loss, f1_macro")

    
    def predict(self, mode: str = "OSA", *args: tuple) -> tuple[NDArray, NDArray]:
        if mode == "OSA":
            return mimo_OSA(self, *args)
        
        if mode == "FreeRun":
            return mimo_FreeRun(self, *args)
        
        if mode == "MShooting":
            return mimo_MShooting(self, *args)
        
        else:
            raise Exception("Choose a mode between: OSA, FreeRun, MShooting")

    
    def _mape(self, yd: NDArray, yp: NDArray) -> float:
        """MAPE Calculate in the array form."""
        diff = np.abs(yd - yp)
        denominator = np.abs(np.max(yd, axis=0) - np.min(yd, axis=0))
        
        safe_denominator = np.where(denominator == 0, np.nan, denominator)
        mape_per_output = 100 * np.nanmean(diff / safe_denominator, axis=0)
        return np.nanmean(mape_per_output) 

    
    def _compute_metric_per_output(self, yd: NDArray, yp: NDArray, metric_func: callable) -> float:
        """Apply a metric (MSE, RMSE) in each output and return the mean"""
        return np.mean([metric_func(yd[:, i], yp[:, i]) for i in range(yd.shape[1])])
        # return max([metric_func(yd[:, i], yp[:, i]) for i in range(yd.shape[1])])

    
    def score(self, yd: NDArray, yp: NDArray, mode: str ="MSE"):
        """Calculate the error metric choosed (MSE, MAPE, RMSE)."""
        if mode not in ["MSE", "NMSE", "MAPE", "RMSE"]:
            raise ValueError("Choose a valid metric: MSE, NMSE, MAPE or RMSE")
        
        if mode == "MSE":
            return self._compute_metric_per_output(yd, yp, mean_squared_error)
        
        if mode == "MAPE":
            return self._mape(yd, yp)
        
        if mode == "RMSE":
            return self._compute_metric_per_output(yd, yp, lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)))
        
        if mode == "NMSE":
            return self._compute_metric_per_output(yd, yp, self._nmse_variance)
        
    
    def _nmse_variance(self, y_true, y_pred) -> float:

        mse = mean_squared_error(y_true, y_pred)
        variance = np.var(y_true)
        return mse / variance if variance != 0 else mse


    @abstractmethod
    def model2List(self):
        pass


    def parse_tree(self, tree: gp.PrimitiveTree, i: int = 0, lag_acc: int = 0) -> tuple[str, int]:
        node = tree[i]

        if hasattr(node, "name") and node.name.startswith("q"):
            k = int(node.name[1:])
            return self.parse_tree(tree, i + 1, lag_acc + k)

        elif isinstance(node, gp.Primitive):
            args = []
            next_i = i + 1

            for _ in range(node.arity):
                arg_str, next_i = self.parse_tree(tree, next_i, lag_acc)
                args.append(arg_str)

            if node.name == "mul":
                return " * ".join(args), next_i

            elif node.name == "subtraction":
                return f"({args[0]} - {args[1]})", next_i

            elif node.name == "sign":
                return f"sign({args[0]} - {args[1]})", next_i

            else:
                return f"{node.name}({', '.join(args)})", next_i

        elif isinstance(node, gp.Terminal):
            var = node.value           
            base = var

            if base.startswith('u'):
                lag = lag_acc
            elif base.startswith('y'):
                lag = lag_acc + 1
            else:
                lag = lag_acc

            return f"{base}[k-{lag}]" if lag > 0 else f"{base}[k]", i + 1

        else:
            return "", i + 1
        

    def to_equation(self) -> str:

        string = ''

        for k, program in enumerate(self):
            string += f'Output {k+1}:\n\n'

            # Bias da saída k
            string += f'{self.theta[k][0]:.5e} + \n'

            for j, tree in enumerate(program):
                expr, _ = self.parse_tree(tree)
                string += f'{self.theta[k][j+1]:.5e} * {expr} + \n'

            string += '\n'

        return string


class Element(object):
    def __init__(self,
                 weights: tuple = (-1,),
                 nDelays: int | List[int] = 3,
                 nInputs: int = 1,
                 nOutputs: int = 1,
                 nTerms: int = 10,
                 maxHeight: int = 5,
                 mode: str = "MISO",
                 single_delay_only: bool = False,
                 operators: List[str] = ['mul', 'subtraction', 'sign', 'add', 'tanh', 'div', 'sin']):
        
        self._pset: gp.PrimitiveSet = None
        self._toolbox: base.Toolbox = None
        self._mode = mode.upper()
        self._single_delay_only = single_delay_only

        if nInputs == 1 and nOutputs == 1:
            self._mode = "SISO"
        
        if self._mode == "FIR":
            self._nVar = nInputs

        elif self._mode == "SISO":
            self._nVar = 2 

        else:
            self._nVar = nInputs + nOutputs
        
        if type(nDelays) is int:
            self._delays = np.arange(1, nDelays + 1)
        elif type(nDelays) is list:
            self._delays = nDelays

        self.operators = operators
        self.primitives_sets = {
            "mul": (operator.mul, 2),
            "sign": (sign, 2),
            "subtraction": (operator.sub, 2),
            "add": (operator.add, 2),
            "tanh": (tanh, 2),
            "div": (div, 2),
            "sin": (sin, 1)
        }

        self.iniciatePrimitivesSets()

        self._weights = weights
        self.nTerms = nTerms
        self._nOutputs = nOutputs
        self._maxHeight = maxHeight     

        creator.create("Program", gp.PrimitiveTree, fitness=None, pset=self.pset)
        creator.create("FitnessMin", base.Fitness, weights=self._weights)

        if self._mode == "MISO":
            creator.create("Individual", IndividualMISO, fitness=creator.FitnessMin)

        elif self._mode == "MIMO":
            creator.create("Individual", IndividualMIMO, fitness=creator.FitnessMin)

        elif self._mode == "SISO":
            creator.create("Individual", IndividualSISO, fitness=creator.FitnessMin)

        elif self._mode == "FIR":
            if nOutputs == 1:
                creator.create("Individual", IndividualFIR, fitness=creator.FitnessMin)

            else:
                creator.create("Individual", IndividualFIRMIMO, fitness=creator.FitnessMin)
        else:
            raise Exception("Choose a mode between:\n" + "MISO, MIMO, FIR or SISO")
        self.iniciateToolbox()


    def iniciatePrimitivesSets(self) -> None:
        
        self._pset = gp.PrimitiveSet("main", self._nVar)
        
        for operator in self.operators:

            if operator in self.primitives_sets.keys():
                func, arity = self.primitives_sets[operator]
                self._pset.addPrimitive(func, arity, name=operator)
            
            else:
                raise ValueError(f"Operator '{operator}' is not defined in primitives_sets.")
        
        if not self._single_delay_only:
            delays = [partial(_roll, i=i) for i in self._delays]

            for i, roll in zip(self._delays, delays):
                self._pset.addPrimitive(roll, 1, name=f'q{i}')
        
        else:
            self._pset.addPrimitive(partial(_roll, i=1), 1, name=f'q1')


    def iniciateToolbox(self) -> None:

        def generate_terminal_only() -> gp.PrimitiveTree:
            """Retorna uma árvore com apenas um nó terminal escolhido aleatoriamente"""
            
            terminal_index = random.randint(1, self._nVar)

            if self._mode == "FIR":
                terminal_name = f"u{terminal_index}"
            
            else:
                
                if terminal_index <= (self._nVar - self._nOutputs):
                    terminal_name = f"u{terminal_index}"

                else:
                    terminal_name = f"y{terminal_index}"

            terminal_node = self._pset.mapping[terminal_name]
            return gp.PrimitiveTree([terminal_node])
        
        self._toolbox = base.Toolbox()
        # self._toolbox.register("_expr", generate_terminal_only)
        self._toolbox.register("_expr", self.genGrowLimitedNodes, pset=self._pset, min_depth=0, max_depth=self._maxHeight, max_nodes=20)
        # self._toolbox.register("_expr", gp.genHalfAndHalf, pset=self._pset, min_=0, max_=self._maxHeight)
        self._toolbox.register("_program", tools.initIterate, creator.Program, self._toolbox._expr)

        if self._mode == "SISO":
            
            creator.create("Individual", IndividualSISO, fitness=creator.FitnessMin)
            self._toolbox.register("individual", tools.initRepeat, creator.Individual, self._toolbox._program, self.nTerms)
        
        elif self._mode == "MISO" or (self._mode == "FIR" and self._nOutputs == 1):
        # Para MISO ou FIR SISO (1 saída)
            self._toolbox.register("individual", tools.initRepeat, creator.Individual, self._toolbox._program, self.nTerms)
        
        elif self._mode == "MIMO" or (self._mode == "FIR" and self._nOutputs > 1):
            self._toolbox.register("_outputs", tools.initRepeat, list, self._toolbox._program, self.nTerms)
            self._toolbox.register("individual", tools.initRepeat, creator.Individual, self._toolbox._outputs, self._nOutputs)
        
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)


    def genGrowLimitedNodes(self, pset: gp.PrimitiveSet, min_depth: int, max_depth: int, max_nodes: int) -> List:
        """Versão do genGrow com limite de nós"""
        while True:
            expr = gp.genHalfAndHalf(pset, min_depth, max_depth)
            tree = gp.PrimitiveTree(expr)
            if len(tree) <= max_nodes:
                return expr


    @property
    def pset(self) -> gp.PrimitiveSet:
        return self._pset
    

    @property
    def toolbox(self) -> base.Toolbox:
        return self._toolbox


    def renameArguments(self, dictionary: dict = {'ARG0': 'y', 'ARG1': 'u'}) -> None:
        self._pset.renameArguments(**dictionary)


    def addPrimitive(self, *args: tuple) -> None:
        self._pset.addPrimitive(*args)


    def buildModelFromList(self, listString: List[str]) -> Individual:
        model = creator.Individual()

        if self._mode == "SISO":
            for string in listString:
                model.append(gp.PrimitiveTree.from_string(string, self.pset))
    
        if self._mode == "MISO" or self._mode == "FIR" and self._nOutputs == 1:
            for string in listString:
                model.append(gp.PrimitiveTree.from_string(string, self.pset))
        
        if self._mode == "MIMO" or (self._mode == "FIR" and self._nOutputs > 1):
            for out in listString:
                aux = [gp.PrimitiveTree.from_string(string, self.pset) for string in out]
                model.append(aux)
        return model


    def buildRandomModel(self) -> Individual:
        return self._toolbox.individual()


    def compileModel(self, model) -> None:
        if self._mode == 'SISO':
            model.funcs = [gp.compile(tree, self.pset) for tree in model]

        elif self._mode == 'MISO' or self._mode == "FIR" and self._nOutputs == 1:
            model.funcs = [gp.compile(tree, self.pset) for tree in model]

        elif self._mode == 'MIMO' or (self._mode == "FIR" and self._nOutputs > 1):
            model.funcs = [[gp.compile(tree, self.pset) for tree in out] for out in model]
            # model.funcs = [[self._compile_to_function(tree, self.pset) for tree in out] for out in model]

        self._setModelLagMax(model)


    def _compile_to_function(self, expr, pset: gp.PrimitiveSet) -> dict:
        code = str(expr)

        if len(pset.arguments) > 0:
            args = ",".join(arg for arg in pset.arguments)
            code = f"def generated_function({args}):\n    return {code}"

        local_scope = {}
        try:
            exec(code, pset.context, local_scope)
            return local_scope["generated_function"]
        
        except MemoryError:
            _, _, traceback = sys.exc_info()
            raise MemoryError("DEAP: Error in tree evaluation: Python cannot evaluate a tree higher than 90. "
                "To avoid this problem, you should use bloat control on your operators. "
                "See the DEAP documentation for more information. DEAP will now abort."
            ).with_traceback(traceback)


    def _setModelLagMax(self, model):
        def checkbranch(branch):
            if branch == []: return
            if branch[-1][2] == branch[-1][1]:
                del branch[-1]
                if branch == []: return
                branch[-1][2] += 1
                return checkbranch(branch)
            else:
                return

        def checkOut(output):
            treelags = []
            for tree in output:
                i = 0
                lagMax = 0
                branches = []
                count = 0
                while i < len(tree):
                    if re.search("q\d", tree[i].name):
                        count += int(tree[i].name[1:])
                    elif type(tree[i]) == gp.Primitive:
                        branches.append([count, tree[i].arity, 0])
                        count = 0
                    elif type(tree[i]) == gp.Terminal:
                        if branches == []:
                            lag = count
                            model.terminals += str(tree[i].value) + '[i-%d] * ' % (count + 1)
                        else:
                            branches[-1][2] += 1
                            lag = count + sum([item[0] for item in branches])
                            model.terminals += tree[i].value + '[i-%d] * ' % (lag + 1)
                        if lag > lagMax:
                            lagMax = lag
                        count = 0
                        checkbranch(branches)
                    i += 1
                treelags.append(lagMax)
                model.terminals += '\n'
            model.terminals += '\n'
            return max(treelags)

        if self._mode == "SISO":
            model.lagMax = checkOut(model)
        
        elif self._mode == "MISO" or (self._mode == "FIR" and self._nOutputs == 1):
            model.lagMax = checkOut(model)
        
        elif self._mode == "MIMO" or (self._mode == "FIR" and self._nOutputs > 1):
            aux = []
            for i, out in enumerate(model):
                model.terminals += 'Output %d:\n\n' % (i + 1)
                aux.append(checkOut(out))
            model.lagMax = max(aux)


    def save(self, filename: str, dictionary: dict) -> None:
        
        with open(filename, 'wb') as f:
            pickle.dump(dictionary, f)
            f.close()

    def load(self, filename: str) -> dict:
        
        with open(filename, 'rb') as f:
            o = pickle.load(f)
            f.close()
            return o



def theta_miso(p, yd):
    return np.linalg.inv(p.T @ p) @ p.T @ yd
    # return np.linalg.lstsq(p, yd, rcond=None)[0]


def theta_mimo(p, yd):
    # return np.dot(np.dot(np.linalg.inv(np.dot(p.T, p)), p.T), yd)
    return np.linalg.lstsq(p, yd, rcond=None)[0]
    # return cp.linalg.lstsq(cp.asarray(p), cp.asarray(yd), rcond=None)[0]


def theta_fir(p, yd):
    return np.linalg.inv(p.T @ p) @ p.T @ yd


class IndividualSISO(Individual):
    
    def __init__(self, data: str = []):
        super().__init__(data)
    
    def predict(self, mode: str = "OSA", *args)  -> tuple[NDArray, NDArray]:
        if mode == "OSA":
            return miso_OSA(self, *args)
        
        if mode == "FreeRun":
            return miso_FreeRun(self, *args)
        
        if mode == "MShooting":
            return miso_MShooting(self, *args)
        
        else:
            raise Exception("Choose a mode between: OSA, FreeRun, MShooting")
    

    def makeRegressors(self, y: NDArray, u: NDArray) -> NDArray:

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if len(u.shape) == 1:
            u = u.reshape(-1, 1)
                
        N = y.shape[0]
        n_samples = N - self.lagMax

        if n_samples <= 0:
            raise ValueError("Not enough samples for given lagMax.")

        y_reg = y[self.lagMax - 1 : N - 1]   # y[k-1]
        u_reg = u[self.lagMax : N]           # u[k]

        listV = [y_reg.reshape(-1, 1), u_reg.reshape(-1, 1)]

        p = np.ones((n_samples, len(self) + 1))
        # p = np.ones((n_samples, len(self)))

        for i in range(len(self)):
            func = self.funcs[i]
            out = func(*listV).reshape(-1)

            if len(out) != n_samples:
                raise ValueError(f"Inconsistent regressor size in term {i}: expected {n_samples}, got {len(out)}")

            p[:, i + 1] = out
            # p[:, i] = out
                    
        return p
    
    
    def leastSquares(self, y: NDArray, u: NDArray) -> NDArray:
        p = self.makeRegressors(y, u)

        yd = y[self.lagMax:]

        self.theta = np.linalg.lstsq(p, yd, rcond=None)[0]
        if len(self.theta.shape) == 1:
            self.theta = self.theta.reshape(-1, 1)
        return self.theta
    

    def __str__(self) -> str:
        string = ''.join('%s\n' * len(self)) % tuple([str(tree) for tree in self])
        return '1\n' + string
    

    def model2List(self) -> List[str]:
        return [str(tree) for tree in self]
    
    
    def to_equation(self) -> str:

        string = ''
        # Bias
        string += f"{self.theta[0][0]:.5e} + \n"

        for j, tree in enumerate(self):
            expr, _ = self.parse_tree(tree)
            string += f"{self.theta[j+1][0]:.5e} * {expr} + \n"
            # string += f"{self.theta[j][0]:.5e} * {expr} + \n"

        return string


class IndividualMISO(Individual):

    def __init__(self, data: str = []):
        super().__init__(data)

    def predict(self, mode: str = "OSA", *args: tuple) -> tuple[NDArray, NDArray]:
        if mode == "OSA":
            return miso_OSA(self, *args)
        
        if mode == "FreeRun":
            return miso_FreeRun(self, *args)
        
        if mode == "MShooting":
            return miso_MShooting(self, *args)
        
        else:
            raise Exception("Choose a mode between: OSA, FreeRun, MShooting")

    # def predict_proba(self, mode="INSTANT", *args):
    #     """Predição de probabilidades para classificação"""
    #     def one_hot_argmax(x):

    #         result = np.zeros_like(x)
    #         result[np.argmax(x)] = 1
            
    #         return result

    #     def softmax(x):
    #         """Calcula softmax para cada linha do vetor de entrada x."""
    #         # Subtrair o máximo melhora a estabilidade numérica (evita overflow)
    #         e_x = np.exp(x - np.max(x))
    #         return e_x / e_x.sum()
        
    #     if mode == "INSTANT":
    #         X_regressors, y_true = miso_CLASSIFY(self, *args)
    #         # y_true = y_true[:len(y_true)-1] # para a classificação FIR precisa desse termo. Já para a MIMO não.
    #     # elif mode == "FreeRun":
    #     #     X_regressors, y_true = mimo_FreeRun(self, *args)
        
    #     elif mode == "MShooting":
    #         X_regressors, y_true = miso_MShooting(self, *args)
        
    #     else:
    #         raise Exception("Choose a mode between: INSTANT")
        
    #     if self.logistic_model is not None:            
    #         # probabilities = np.array([one_hot_argmax(x) for x in X_regressors])
    #         pred_one_hot_encode = np.array([one_hot_argmax(softmax(x)) for x in X_regressors])
    #         # pred_labels = np.array([np.argmax(prob) for prob in pred_one_hot_encode]).astype(np.float64)
    #         return pred_one_hot_encode, y_true
        
    #     else:
    #         raise Exception("Logistic regression model not trained!")



    def makeRegressors(self, y: NDArray, u: NDArray) -> NDArray:
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if len(u.shape) == 1:
            u = u.reshape(-1, 1)

        if y.shape[1] > 1:
            raise Exception('Wrong number of outputs. The algorithm is set',
                            'for a single output')
        is_classification = bool(getattr(self, "logistic_model", False))

        listV = []
        if is_classification:
            listV = [y.reshape(-1, 1)]

        else:
            listV = [y[:-1].reshape(-1, 1)]
        
        for v in u.T:
            if is_classification:
                listV.append(v.reshape(-1, 1))

            else:
                listV.append(v[:-1].reshape(-1, 1))
        
        if is_classification:
            n_samples = y.shape[0] - self.lagMax

        else:
            n_samples = y.shape[0] - self.lagMax - 1
            
        p = np.ones((n_samples, len(self) + 1))

        for i in range(len(self)):
            func = self.funcs[i]
            out = func(*listV)
            p[:, i + 1] = out.reshape(-1)[self.lagMax:]

        return p


    def leastSquares(self, y: NDArray, u: NDArray) -> NDArray:
        '''
        The leastSquare(y,u) function implements the Least Squares method
        for parameter estimation.
        
        The arguments are the output y and the inputs u, in which each entry 
        must be in column formm.
        '''
        p = self.makeRegressors(y, u)
        # if np.linalg.cond(p, -2) < 1e-10:
        #     raise np.linalg.LinAlgError('Ill conditioned regressors matrix!')
        
        is_classification = bool(getattr(self, "logistic_model", False))
        offset = 0 if is_classification else 1  # <-- chave do alinhamento

        yd = y[self.lagMax + offset:]
        # self.theta = np.linalg.inv(p.T @ p) @ p.T @ yd
        self.theta = theta_miso(p, yd)
        if len(self.theta.shape) == 1:
            self.theta = self.theta.reshape(-1, 1)
        return self.theta


    def __str__(self) -> str:
        string = ''.join('%s\n' * len(self)) % tuple([str(tree) for tree in self])
        return '1\n' + string


    def model2List(self) -> List[str]:
        listString = []
        for tree in self:
            listString.append(str(tree))
        # return listString
        return [str(tree) for tree in self]


class IndividualMIMO(Individual):
    def __init__(self, data: str = []):
        super().__init__(data)

    
    def makeRegressors(self, y: NDArray, u: NDArray) -> NDArray:
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if len(u.shape) == 1:
            u = u.reshape(-1, 1)

        if y.shape[1] == 1:
            raise Exception('Wrong number of outputs. The algorithm is set for multiple outputs')

        listV = []
        for v in y.T:
            listV.append(v.reshape(-1, 1))
        
        for v in u.T:
            listV.append(v.reshape(-1, 1))

        P = []
        for o in range(len(self)):

            n_samples = y.shape[0] - self.lagMax
            p = np.ones((n_samples, len(self[o]) + 1))

            for i in range(len(self[o])):
                func = self.funcs[o][i]
                out = func(*listV)
                p[:, i + 1] = out.reshape(-1)[self.lagMax:] #TODO: GARGALO!!!!
            P.append(p)
        return P


    def leastSquares(self, y: NDArray, u: NDArray) -> NDArray:
        """
        LS para MIMO.
        Em classificação, y_true deve alinhar com P (N - lagMax).
        Em regressão OSA, mantém lagMax+1.
        """
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if len(u.shape) == 1:
            u = u.reshape(-1, 1)

        P = self.makeRegressors(y, u)

        # is_classification = bool(getattr(self, "logistic_model", False))
        # offset = 0 if is_classification else 1  # <-- chave do alinhamento

        y_slice = y[self.lagMax:, :]   # (N-lagMax) ou (N-lagMax-1)

        self.theta = np.array([theta_mimo(P[o], y_slice[:, o]) for o in range(len(P))])
        return self.theta


    def __str__(self) -> str:
        string = ''
        i = 1
        for out in self:
            string += 'Output %d:\n\n1\n' % (i)
            for tree in out:
                string += str(tree) + '\n'
            i += 1
            string += '\n'
        return string


    def model2List(self) -> List[List[str]]:
        return [[str(tree) for tree in out] for out in self]


class IndividualFIR(Individual):

    def __init__(self, data: str = []):
        super().__init__(data)

    # def predict(self, mode="OSA", *args):
    #     if mode == "OSA":
    #         return miso_OSA(self, *args)
    #     if mode == "FreeRun":
    #         return miso_FreeRun(self, *args)
    #     if mode == "MShooting":
    #         return miso_MShooting(self, *args)
    #     else:
    #         raise Exception("Choose a mode between: OSA, FreeRun, MShooting")

    # def makeRegressors(self, y, u):
    #     if len(u.shape) == 1:
    #         u = u.reshape(-1, 1)

    #     listV = []
    #     for v in u.T:
    #         listV.append(v[:-1].reshape(-1, 1))

    #     p = np.ones((u.shape[0] - self.lagMax - 1, len(self) + 1))

    #     for i in range(len(self)):
    #         func = self.funcs[i]
    #         out = func(*listV)
    #         p[:, i + 1] = out.reshape(-1)[self.lagMax:]
    #     return p

    # def leastSquares(self, y, u):
    #     '''
    #     The leastSquare(y,u) function implements the Least Squares method
    #     for parameter estimation.
        
    #     The arguments are the output y and the inputs u, in which each entry 
    #     must be in column formm.
    #     '''
    #     p = self.makeRegressors(y, u)
    #     if np.linalg.cond(p, -2) < 1e-10:
    #         raise np.linalg.LinAlgError(
    #             'Ill conditioned regressors matrix!')
    #     yd = y[self.lagMax + 1:]
    #     # self.theta = np.linalg.inv(p.T @ p) @ p.T @ yd
    #     self.theta = theta_fir(p, yd)
    #     if len(self.theta.shape) == 1:
    #         self.theta = self.theta.reshape(-1, 1)
    #     return self.theta

    def predict(self, mode: str ="OSA", *args: tuple) -> tuple[NDArray, NDArray]:
        if mode == "OSA":
            return miso_OSA(self, *args)
        
        if mode == "INSTANT":
            return miso_FIR_INSTANT(self, *args)
        
        if mode == "FreeRun":
            return miso_FreeRun(self, *args)
        
        if mode == "MShooting":
            return miso_MShooting(self, *args)
        
        else:
            raise Exception("Choose a mode between: OSA, INSTANT, FreeRun, MShooting")


    def makeRegressors(self, y: NDArray, u: NDArray, align: str = "OSA") -> NDArray:
        if len(u.shape) == 1:
            u = u.reshape(-1, 1)

        listV = []
        if align == 'INSTANT': 
            for v in u.T:
                listV.append(v.reshape(-1, 1))
            n_samples = u.shape[0] - self.lagMax
        
        else:
            for v in u.T:
                listV.append(v[:-1].reshape(-1, 1))
            n_samples = u.shape[0] - self.lagMax - 1

        if n_samples <= 0:
            raise ValueError("Not enough samples for the chosen lagMax/alignment.")

        p = np.ones((n_samples, len(self) + 1))

        for i in range(len(self)):
            func = self.funcs[i]
            out = func(*listV)
            p[:, i + 1] = out.reshape(-1)[self.lagMax:]

        return p


    def leastSquares(self, y: NDArray, u: NDArray, align: str ="OSA") -> NDArray:
        """
        FIR LS com alinhamento consistente com a predição.
        """
        p = self.makeRegressors(y, u, align=align)

        if np.linalg.cond(p, -2) < 1e-10:
            raise np.linalg.LinAlgError('Ill conditioned regressors matrix!')

        if align == "INSTANT":
            yd = y[self.lagMax:]

        else:
            yd = y[self.lagMax + 1:]

        self.theta = theta_fir(p, yd)
        if len(self.theta.shape) == 1:
            self.theta = self.theta.reshape(-1, 1)

        return self.theta

    def __str__(self) -> str:
        string = ''.join('%s\n' * len(self)) % tuple([str(tree) for tree in self])
        return '1\n' + string

    def model2List(self) -> List[str]:
        return [str(tree) for tree in self]

class IndividualFIRMIMO(Individual):
    def __init__(self, data: list = []):
        super().__init__(data)

    # def makeRegressors(self, y, u):
    #     if len(u.shape) == 1:
    #         u = u.reshape(-1, 1)

    #     listV = []
    #     for v in u.T:
    #         listV.append(v[:-1].reshape(-1, 1))

    #     P = []
    #     for o in range(len(self)):  # Para cada saída
    #         p = np.ones((u.shape[0] - self.lagMax - 1, len(self[o]) + 1))
    #         for i in range(len(self[o])):  # Para cada termo da saída
    #             func = self.funcs[o][i]
    #             out = func(*listV)
    #             p[:, i + 1] = out.reshape(-1)[self.lagMax:]
    #         P.append(p)
    #     return P

    # def leastSquares(self, y, u):
    #     P = self.makeRegressors(y, u)
    #     self.theta = np.array([theta_mimo(P[o], y[self.lagMax + 1:, o]) for o in range(len(P))])
    #     return self.theta
    
    # def predict(self, mode="OSA", *args):
    #     if mode == "OSA":
    #         return mimo_OSA(self, *args)
    #     if mode == "FreeRun":
    #         return mimo_FIR_FreeRun(self, *args)
    #     if mode == "MShooting":
    #         return mimo_FIR_MShooting(self, *args)
    #     else:
    #         raise Exception("Choose a mode between: OSA, FreeRun, MShooting")


    def makeRegressors(self, y: NDArray, u: NDArray) -> List[NDArray]:
        if len(u.shape) == 1:
            u = u.reshape(-1, 1)

        listV = []
        for v in u.T:
            listV.append(v.reshape(-1, 1))
        n_samples = u.shape[0] - self.lagMax

        if n_samples <= 0:
            raise ValueError("Not enough samples for the chosen lagMax/alignment.")

        P = []
        for o in range(len(self)):  # Para cada saída
            p = np.ones((n_samples, len(self[o]) + 1))
            
            for i in range(len(self[o])):  # Para cada termo da saída
                func = self.funcs[o][i]
                out = func(*listV)
                p[:, i + 1] = out.reshape(-1)[self.lagMax:]
            
            P.append(p)
        return P


    def leastSquares(self, y: NDArray, u: NDArray) -> NDArray:
        P = self.makeRegressors(y, u)

        y_slice = y[self.lagMax:, :]

        self.theta = np.array([theta_mimo(P[o], y_slice[:, o]) for o in range(len(P))])
        return self.theta


    def predict(self, mode: str ="OSA", *args: tuple) -> tuple[NDArray, NDArray]:
        if mode == "OSA":
            return mimo_OSA(self, *args)
        if mode == "INSTANT":
            return mimo_FIR_INSTANT(self, *args)
        if mode == "FreeRun":
            return mimo_FIR_FreeRun(self, *args)
        if mode == "MShooting":
            return mimo_FIR_MShooting(self, *args)
        else:
            raise Exception("Choose a mode between: OSA, INSTANT, FreeRun, MShooting")


    def __str__(self) -> str:
        string = ''
        i = 1
        for out in self:
            string += 'Output %d:\n\n1\n' % (i)
            for tree in out:
                string += str(tree) + '\n'
            i += 1
            string += '\n'
        return string

    def model2List(self) -> List[List[str]]:
        return [[str(tree) for tree in out] for out in self]
    