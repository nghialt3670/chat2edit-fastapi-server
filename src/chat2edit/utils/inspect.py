import ast
from typing import Any, List


def extract_args_from_func_call_str(func_call_str: str) -> List[Any]:
    # Parse the function call string into an AST node
    func_call_node = ast.parse(func_call_str.strip()).body[0].value

    # Initialize a list to store the arguments
    args = []

    # Function to evaluate an AST node
    def eval_ast_node(node):
        if isinstance(node, ast.Constant):  # For constant values (e.g., 4, 'string')
            return node.value
        elif isinstance(node, ast.Name):  # For variable names
            return node.id
        elif isinstance(node, ast.Dict):  # For dictionaries
            return {
                eval_ast_node(k): eval_ast_node(v)
                for k, v in zip(node.keys, node.values)
            }
        elif isinstance(node, ast.List):  # For lists
            return [eval_ast_node(e) for e in node.elts]
        elif isinstance(node, ast.Tuple):  # For tuples
            return tuple(eval_ast_node(e) for e in node.elts)
        elif isinstance(node, ast.Call):  # For nested function calls (if needed)
            return ast.unparse(node) if hasattr(ast, "unparse") else ast.dump(node)
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")

    # Extract positional arguments
    for arg in func_call_node.args:
        args.append(eval_ast_node(arg))

    # Extract keyword arguments
    for kw in func_call_node.keywords:
        value = eval_ast_node(kw.value)
        args.append(f"{kw.arg}={value}")

    return args
