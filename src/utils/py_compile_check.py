"""
Python compilation checking utilities for SAQ.
"""
import ast
import sys
import traceback
from typing import Optional, Tuple


def compile_ok(code: str) -> bool:
    """
    Check if Python code compiles without syntax errors.
    
    Args:
        code: Python source code string
        
    Returns:
        bool: True if code compiles successfully, False otherwise
    """
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False
    except Exception:
        # Other compilation errors (indentation, etc.)
        return False


def ast_parse_ok(code: str) -> bool:
    """
    Check if Python code can be parsed into an AST.
    
    Args:
        code: Python source code string
        
    Returns:
        bool: True if code parses successfully, False otherwise
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False


def detailed_compile_check(code: str) -> Tuple[bool, Optional[str]]:
    """
    Check compilation and return detailed error information.
    
    Args:
        code: Python source code string
        
    Returns:
        Tuple[bool, Optional[str]]: (success, error_message)
    """
    try:
        compile(code, '<string>', 'exec')
        return True, None
    except SyntaxError as e:
        error_msg = f"SyntaxError: {e.msg} at line {e.lineno}"
        return False, error_msg
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        return False, error_msg


def extract_function_body(code: str, func_name: str = None) -> str:
    """
    Extract function body from generated code.
    Useful when model generates extra text around the function.
    
    Args:
        code: Generated code string
        func_name: Optional function name to extract (if None, extracts first function)
        
    Returns:
        str: Extracted function code or original code if no function found
    """
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if func_name is None or node.name == func_name:
                    # Extract the function definition
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else len(code.split('\n'))
                    lines = code.split('\n')
                    return '\n'.join(lines[start_line:end_line])
        return code  # Return original if no function found
    except:
        return code  # Return original if parsing fails


if __name__ == "__main__":
    # Test cases
    test_cases = [
        "def hello():\n    print('world')",  # Valid
        "def hello(\n    print('world')",    # Invalid syntax
        "def hello():\nprint('world')",      # Invalid indentation
        "x = 1 + 2",                         # Valid simple
        "x = 1 + ",                          # Invalid
    ]
    
    for i, code in enumerate(test_cases):
        result = compile_ok(code)
        ast_result = ast_parse_ok(code)
        detailed_result, error = detailed_compile_check(code)
        print(f"Test {i+1}: compile={result}, ast={ast_result}, detailed={detailed_result}")
        if error:
            print(f"  Error: {error}")
        print(f"  Code: {repr(code)}")
        print() 