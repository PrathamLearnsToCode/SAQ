"""
Syntax-aware reward functions for SAQ.
"""
import ast
import numpy as np
from typing import Dict, List, Optional, Union, Callable
from enum import Enum
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.py_compile_check import compile_ok, detailed_compile_check
from utils.ast_check import ASTChecker, quick_syntax_check


class RewardType(Enum):
    """Types of syntax rewards available."""
    BINARY_COMPILE = "binary_compile"
    AST_NODE_RATIO = "ast_node_ratio" 
    PARSE_ERROR_RATIO = "parse_error_ratio"
    WEIGHTED_SYNTAX = "weighted_syntax"
    COMPOSITE = "composite"


class SyntaxRewardCalculator:
    """Calculate syntax-aware rewards for generated code."""
    
    def __init__(self, reward_type: str = "binary_compile", use_tree_sitter: bool = True):
        """
        Initialize syntax reward calculator.
        
        Args:
            reward_type: Type of reward to calculate
            use_tree_sitter: Whether to use tree-sitter for AST parsing
        """
        self.reward_type = RewardType(reward_type)
        self.ast_checker = ASTChecker(use_tree_sitter=use_tree_sitter)
        
    def calculate_reward(self, code: str, prompt: str = None) -> float:
        """
        Calculate syntax reward for generated code.
        
        Args:
            code: Generated code string
            prompt: Original prompt (optional, for context)
            
        Returns:
            float: Reward value (typically 0.0 to 1.0)
        """
        if self.reward_type == RewardType.BINARY_COMPILE:
            return self._binary_compile_reward(code)
        elif self.reward_type == RewardType.AST_NODE_RATIO:
            return self._ast_node_ratio_reward(code)
        elif self.reward_type == RewardType.PARSE_ERROR_RATIO:
            return self._parse_error_ratio_reward(code)
        elif self.reward_type == RewardType.WEIGHTED_SYNTAX:
            return self._weighted_syntax_reward(code)
        elif self.reward_type == RewardType.COMPOSITE:
            return self._composite_reward(code, prompt)
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
    
    def _binary_compile_reward(self, code: str) -> float:
        """
        Binary reward with dense fallback to reduce variance.
        Combines sparse binary signal with dense AST signal.
        """
        if compile_ok(code):
            return 0.8  # High reward for compilation
        else:
            # Fallback to partial AST reward to provide dense signal
            return 0.2 * self._ast_node_ratio_reward(code)
    
    def _ast_node_ratio_reward(self, code: str) -> float:
        """
        Reward based on ratio of successfully parsed AST nodes.
        Higher ratio = better syntax structure.
        """
        try:
            # Parse with Python AST
            tree = ast.parse(code)
            total_nodes = sum(1 for _ in ast.walk(tree))
            
            # Count different types of nodes for quality assessment
            function_nodes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            class_nodes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            control_nodes = sum(1 for node in ast.walk(tree) 
                              if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)))
            
            # Calibrated normalization based on HumanEval dataset analysis
            # Average nodes per HumanEval snippet is ~25-30
            avg_expected_nodes = 25.0
            
            # Use logarithmic scaling to avoid saturation
            base_reward = min(1.0, np.log1p(total_nodes) / np.log1p(avg_expected_nodes))
            
            # Structure bonus for meaningful code constructs
            structure_bonus = 0.1 * (function_nodes + class_nodes + control_nodes)
            
            return min(1.0, base_reward + structure_bonus)
            
        except SyntaxError:
            # Partial parsing - try to count valid lines
            lines = code.split('\n')
            valid_lines = 0
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    compile(line, '<string>', 'exec')
                    valid_lines += 1
                except:
                    pass
            
            # Return partial reward based on valid lines
            return min(0.5, valid_lines / max(1, len([l for l in lines if l.strip()])))
        
        except Exception:
            return 0.0
    
    def _parse_error_ratio_reward(self, code: str) -> float:
        """
        Reward based on inverse of parse error ratio.
        Fewer errors = higher reward.
        """
        try:
            # Check compilation
            is_valid, error_msg = detailed_compile_check(code)
            
            if is_valid:
                return 1.0
            
            # Analyze error severity
            lines = code.split('\n')
            total_lines = len([l for l in lines if l.strip()])
            
            if total_lines == 0:
                return 0.0
            
            # Use AST parsing with error localization instead of per-line compile
            # This preserves multi-line context
            try:
                ast.parse(code)
                return 1.0  # No syntax errors
            except SyntaxError as e:
                if e.lineno and total_lines > 0:
                    # Assign error "mass" based on error line position
                    error_impact = min(3, total_lines - e.lineno + 1) / total_lines
                    return max(0.1, 1.0 - error_impact)
                else:
                    return 0.1  # Unknown error location
            
            # Fallback: count problematic lines for other errors
            error_lines = 0
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    compile(line, '<string>', 'exec')
                except:
                    error_lines += 1
            
            error_ratio = error_lines / max(1, total_lines)
            return max(0.1, 1.0 - error_ratio)
            
        except Exception:
            return 0.0
    
    def _weighted_syntax_reward(self, code: str, weights: Dict[str, float] = None) -> float:
        """
        Weighted combination of multiple syntax features.
        Weights are now configurable hyperparameters.
        """
        if weights is None:
            weights = {
                'compile': 0.4,
                'ast_structure': 0.2,
                'parse_error': 0.15,
                'indentation': 0.1,
                'completeness': 0.1,
                'bracket_balance': 0.05
            }
        
        rewards = {}
        
        # Compilation reward (using improved binary reward)
        rewards['compile'] = self._binary_compile_reward(code)
        
        # AST structure reward
        rewards['ast_structure'] = self._ast_node_ratio_reward(code)
        
        # Parse error reward
        rewards['parse_error'] = self._parse_error_ratio_reward(code)
        
        # Indentation consistency reward
        rewards['indentation'] = self._check_indentation_consistency(code)
        
        # Code completeness reward
        rewards['completeness'] = self._check_code_completeness(code)
        
        # Bracket balancing reward
        rewards['bracket_balance'] = self._check_bracket_balance(code)
        
        # Weighted sum
        total_reward = sum(weights.get(key, 0.0) * rewards[key] for key in rewards)
        return min(1.0, max(0.0, total_reward))
    
    def _composite_reward(self, code: str, prompt: str = None, alpha: float = 0.5, beta: float = 0.5) -> float:
        """
        Composite reward combining dense and sparse signals.
        Uses configurable mixing coefficients alpha and beta.
        """
        # Dense reward component (AST-based)
        dense_reward = self._ast_node_ratio_reward(code)
        
        # Sparse reward component (compilation)
        sparse_reward = 1.0 if compile_ok(code) else 0.0
        
        # Parse error component for additional density
        parse_reward = self._parse_error_ratio_reward(code)
        
        # Balanced combination: alpha * dense + beta * sparse + (1-alpha-beta) * parse
        gamma = max(0.0, 1.0 - alpha - beta)
        composite = (alpha * dense_reward + 
                    beta * sparse_reward + 
                    gamma * parse_reward)
        
        return min(1.0, max(0.0, composite))
    
    def _check_indentation_consistency(self, code: str) -> float:
        """Check indentation consistency in code."""
        lines = code.split('\n')
        indentations = []
        
        for line in lines:
            if line.strip():  # Skip empty lines
                indent = len(line) - len(line.lstrip())
                indentations.append(indent)
        
        if not indentations:
            return 1.0
        
        # Check if indentations follow Python conventions (multiples of 4)
        consistent_indents = sum(1 for indent in indentations if indent % 4 == 0)
        consistency_ratio = consistent_indents / len(indentations)
        
        return consistency_ratio
    
    def _check_code_completeness(self, code: str) -> float:
        """Check if code appears complete (has function definitions, proper structure)."""
        try:
            tree = ast.parse(code)
            
            # Check for function definitions
            has_functions = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
            
            # Check for return statements in functions
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            complete_functions = 0
            
            for func in functions:
                has_return = any(isinstance(node, ast.Return) for node in ast.walk(func))
                if has_return:
                    complete_functions += 1
            
            if not functions:
                # If no functions, check for basic completeness
                has_assignments = any(isinstance(node, ast.Assign) for node in ast.walk(tree))
                return 0.5 if has_assignments else 0.2
            
            # Return ratio of complete functions
            return complete_functions / len(functions)
            
        except:
            return 0.0
    
    def _check_bracket_balance(self, code: str) -> float:
        """Check bracket/parentheses balancing score."""
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        total_brackets = 0
        matched_brackets = 0
        
        for char in code:
            if char in brackets:
                stack.append(char)
                total_brackets += 1
            elif char in brackets.values():
                total_brackets += 1
                if stack and brackets[stack[-1]] == char:
                    stack.pop()
                    matched_brackets += 2  # Count both opening and closing
                else:
                    # Unmatched closing bracket
                    pass
        
        if total_brackets == 0:
            return 1.0  # No brackets to balance
        
        return matched_brackets / total_brackets
    
    def _check_name_errors(self, code: str) -> float:
        """
        Check for name/import errors by attempting execution in sandbox.
        Returns reward based on absence of NameError/ImportError.
        """
        try:
            # Compile first
            compiled_code = compile(code, '<string>', 'exec')
            
            # Try to execute in restricted namespace
            safe_globals = {
                '__builtins__': {
                    'len': len, 'range': range, 'enumerate': enumerate,
                    'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted,
                    'sum': sum, 'max': max, 'min': min, 'abs': abs,
                    'int': int, 'float': float, 'str': str, 'bool': bool,
                    'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
                    'print': lambda *args, **kwargs: None,  # Silent print
                }
            }
            safe_locals = {}
            
            exec(compiled_code, safe_globals, safe_locals)
            return 1.0  # No name/import errors
            
        except (NameError, ImportError):
            return 0.3  # Partial penalty for name errors
        except Exception:
            return 0.7  # Other runtime errors get smaller penalty
    
    def _calculate_ast_distance(self, code: str, reference_code: str = None) -> float:
        """
        Calculate AST-based similarity to reference code (if available).
        This is useful when you have ground truth solutions.
        """
        if not reference_code:
            return 1.0  # No reference to compare against
        
        try:
            # Parse both codes
            tree1 = ast.parse(code)
            tree2 = ast.parse(reference_code)
            
            # Simple AST similarity based on node types
            nodes1 = [type(node).__name__ for node in ast.walk(tree1)]
            nodes2 = [type(node).__name__ for node in ast.walk(tree2)]
            
            # Calculate Jaccard similarity
            set1, set2 = set(nodes1), set(nodes2)
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            if union == 0:
                return 1.0
            
            jaccard_sim = intersection / union
            return jaccard_sim
            
        except:
            return 0.0


def calculate_batch_rewards(
    codes: List[str], 
    prompts: List[str] = None,
    reward_type: str = "binary_compile",
    use_tree_sitter: bool = True
) -> List[float]:
    """
    Calculate rewards for a batch of generated codes.
    
    Args:
        codes: List of generated code strings
        prompts: List of original prompts (optional)
        reward_type: Type of reward to calculate
        use_tree_sitter: Whether to use tree-sitter
        
    Returns:
        List[float]: Reward values for each code
    """
    calculator = SyntaxRewardCalculator(reward_type, use_tree_sitter)
    
    rewards = []
    for i, code in enumerate(codes):
        prompt = prompts[i] if prompts else None
        reward = calculator.calculate_reward(code, prompt)
        rewards.append(reward)
    
    return rewards


def get_reward_statistics(rewards: List[float]) -> Dict[str, float]:
    """Get statistics for a list of rewards."""
    rewards_array = np.array(rewards)
    
    return {
        'mean': float(np.mean(rewards_array)),
        'std': float(np.std(rewards_array)),
        'min': float(np.min(rewards_array)),
        'max': float(np.max(rewards_array)),
        'median': float(np.median(rewards_array)),
        'success_rate': float(np.mean(rewards_array > 0.5))  # Assuming > 0.5 is success
    }


def scale_rewards(rewards: List[float], method: str = "normalize") -> List[float]:
    """
    Scale rewards to improve RL training stability.
    
    Args:
        rewards: List of raw reward values
        method: Scaling method ("normalize", "standardize", "clip")
        
    Returns:
        List[float]: Scaled rewards
    """
    rewards_array = np.array(rewards)
    
    if method == "normalize":
        # Scale to [0, 1] range
        min_r, max_r = rewards_array.min(), rewards_array.max()
        if max_r > min_r:
            return ((rewards_array - min_r) / (max_r - min_r)).tolist()
        else:
            return rewards_array.tolist()
    
    elif method == "standardize":
        # Z-score normalization
        mean_r, std_r = rewards_array.mean(), rewards_array.std()
        if std_r > 0:
            return ((rewards_array - mean_r) / std_r).tolist()
        else:
            return (rewards_array - mean_r).tolist()
    
    elif method == "clip":
        # Clip outliers to reduce variance
        q25, q75 = np.percentile(rewards_array, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        return np.clip(rewards_array, lower_bound, upper_bound).tolist()
    
    else:
        return rewards


class RewardBaseline:
    """
    Running baseline for reward normalization in RL.
    Helps reduce variance in policy gradients.
    """
    
    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.running_mean = 0.0
        self.running_var = 0.0
        self.count = 0
    
    def update(self, rewards: List[float]) -> List[float]:
        """Update baseline and return advantage values."""
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        
        if self.count == 0:
            self.running_mean = batch_mean
            self.running_var = batch_var
        else:
            self.running_mean = self.decay * self.running_mean + (1 - self.decay) * batch_mean
            self.running_var = self.decay * self.running_var + (1 - self.decay) * batch_var
        
        self.count += 1
        
        # Return advantages (reward - baseline)
        advantages = [r - self.running_mean for r in rewards]
        return advantages
    
    def get_baseline(self) -> float:
        """Get current baseline value."""
        return self.running_mean


if __name__ == "__main__":
    # Test the reward system
    test_codes = [
        # Valid complete function
        """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
        """,
        
        # Valid but incomplete
        """
def fibonacci(n):
    if n <= 1:
        # Missing return
        """,
        
        # Syntax error
        """
def fibonacci(n
    if n <= 1:
        return n
        """,
        
        # Empty/minimal
        "x = 1",
        
        # Complex valid code
        """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
        """
    ]
    
    # Test different reward types
    reward_types = ["binary_compile", "ast_node_ratio", "weighted_syntax", "composite"]
    
    for reward_type in reward_types:
        print(f"\n=== {reward_type.upper()} REWARDS ===")
        rewards = calculate_batch_rewards(test_codes, reward_type=reward_type)
        
        for i, (code, reward) in enumerate(zip(test_codes, rewards)):
            code_preview = code.strip().split('\n')[0][:50] + "..."
            print(f"Code {i+1}: {reward:.3f} - {code_preview}")
        
        stats = get_reward_statistics(rewards)
        print(f"Stats: mean={stats['mean']:.3f}, success_rate={stats['success_rate']:.3f}") 