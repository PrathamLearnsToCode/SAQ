"""
Advanced AST checking utilities using tree-sitter for SAQ.
"""
import ast
from typing import Dict, List, Optional, Tuple, Any

try:
    import tree_sitter
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    # Don't print warning by default - will fallback to ast module


class ASTChecker:
    """AST validation and analysis utilities."""
    
    def __init__(self, use_tree_sitter: bool = True):
        self.use_tree_sitter = use_tree_sitter and TREE_SITTER_AVAILABLE
        self.parser = None
        
        if self.use_tree_sitter:
            try:
                # Try to initialize tree-sitter parser
                self._init_tree_sitter()
            except Exception as e:
                print(f"Warning: Failed to initialize tree-sitter: {e}")
                self.use_tree_sitter = False
    
    def _init_tree_sitter(self):
        """Initialize tree-sitter parser for Python."""
        try:
            # Handle different tree-sitter-python versions
            try:
                from tree_sitter_python import language
                self.parser = Parser()
                self.parser.set_language(language())
            except ImportError:
                # Fallback for older versions
                import tree_sitter_python as tspython
                self.parser = Parser()
                self.parser.set_language(tspython.language())
        except Exception as e:
            print(f"Tree-sitter initialization failed: {e}")
            self.use_tree_sitter = False
    
    def check_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Check if code has valid syntax.
        
        Args:
            code: Python source code
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        if self.use_tree_sitter and self.parser:
            return self._check_syntax_tree_sitter(code)
        else:
            return self._check_syntax_ast(code)
    
    def _check_syntax_ast(self, code: str) -> Tuple[bool, Optional[str]]:
        """Check syntax using Python's built-in ast module."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            error_msg = f"SyntaxError at line {e.lineno}: {e.msg}"
            return False, error_msg
        except Exception as e:
            return False, f"ParseError: {str(e)}"
    
    def _check_syntax_tree_sitter(self, code: str) -> Tuple[bool, Optional[str]]:
        """Check syntax using tree-sitter."""
        try:
            tree = self.parser.parse(bytes(code, 'utf8'))
            
            # Check for syntax errors by looking for ERROR nodes
            def has_error(node):
                if node.type == 'ERROR':
                    return True
                for child in node.children:
                    if has_error(child):
                        return True
                return False
            
            if has_error(tree.root_node):
                return False, "Syntax error detected by tree-sitter"
            
            return True, None
        except Exception as e:
            return False, f"Tree-sitter error: {str(e)}"
    
    def analyze_structure(self, code: str) -> Dict[str, Any]:
        """
        Analyze code structure and return metrics.
        
        Args:
            code: Python source code
            
        Returns:
            Dict with structure information
        """
        try:
            tree = ast.parse(code)
            
            analysis = {
                'functions': [],
                'classes': [],
                'imports': [],
                'complexity_score': 0,
                'line_count': len(code.split('\n')),
                'has_docstrings': False
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'line': node.lineno,
                        'args': len(node.args.args),
                        'has_docstring': ast.get_docstring(node) is not None
                    }
                    analysis['functions'].append(func_info)
                    if func_info['has_docstring']:
                        analysis['has_docstrings'] = True
                
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'line': node.lineno,
                        'methods': len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                        'has_docstring': ast.get_docstring(node) is not None
                    }
                    analysis['classes'].append(class_info)
                    if class_info['has_docstring']:
                        analysis['has_docstrings'] = True
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            analysis['imports'].append(alias.name)
                    else:  # ImportFrom
                        module = node.module or ''
                        for alias in node.names:
                            analysis['imports'].append(f"{module}.{alias.name}")
                
                # Simple complexity scoring
                elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                    analysis['complexity_score'] += 1
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def extract_functions(self, code: str) -> List[Dict[str, Any]]:
        """
        Extract all function definitions from code.
        
        Args:
            code: Python source code
            
        Returns:
            List of function information dictionaries
        """
        try:
            tree = ast.parse(code)
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract function source
                    lines = code.split('\n')
                    start_line = node.lineno - 1
                    end_line = getattr(node, 'end_lineno', len(lines)) or len(lines)
                    
                    func_source = '\n'.join(lines[start_line:end_line])
                    
                    func_info = {
                        'name': node.name,
                        'source': func_source,
                        'start_line': node.lineno,
                        'end_line': end_line,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node),
                        'returns': node.returns is not None
                    }
                    functions.append(func_info)
            
            return functions
            
        except Exception as e:
            return [{'error': str(e)}]


# Global instance for convenience
ast_checker = ASTChecker()


def quick_syntax_check(code: str) -> bool:
    """Quick syntax validation using the global checker."""
    is_valid, _ = ast_checker.check_syntax(code)
    return is_valid


def analyze_code_structure(code: str) -> Dict[str, Any]:
    """Quick structure analysis using the global checker."""
    return ast_checker.analyze_structure(code)


if __name__ == "__main__":
    # Test the AST checker
    test_codes = [
        # Valid code
        '''
def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
        ''',
        
        # Invalid syntax
        '''
def broken_func(
    return "missing closing paren"
        ''',
        
        # Complex valid code
        '''
import math
from typing import List

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a: int, b: int) -> int:
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
        '''
    ]
    
    checker = ASTChecker()
    
    for i, code in enumerate(test_codes):
        print(f"\n=== Test {i+1} ===")
        is_valid, error = checker.check_syntax(code)
        print(f"Valid: {is_valid}")
        if error:
            print(f"Error: {error}")
        
        if is_valid:
            analysis = checker.analyze_structure(code)
            print(f"Functions: {len(analysis.get('functions', []))}")
            print(f"Classes: {len(analysis.get('classes', []))}")
            print(f"Imports: {analysis.get('imports', [])}")
            print(f"Complexity: {analysis.get('complexity_score', 0)}") 