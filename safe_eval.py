from __future__ import annotations

import ast
from typing import Any, Mapping


class UnsafeExpressionError(ValueError):
    pass


def evaluate_meson_value(expression: str, symbols: Mapping[str, Any]) -> Any:
    return _eval_meson_node(_parse_expression(expression), symbols)


def evaluate_condition(expression: str, symbols: Mapping[str, Any]) -> bool:
    result = _eval_condition_node(_parse_expression(expression), symbols)
    if not isinstance(result, bool):
        raise UnsafeExpressionError(f"condition did not evaluate to bool: {expression!r}")
    return result


def _parse_expression(expression: str) -> ast.AST:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise UnsafeExpressionError(f"invalid expression: {expression!r}") from e
    return parsed.body


def _resolve_name(name: str, symbols: Mapping[str, Any]) -> Any:
    try:
        return symbols[name]
    except KeyError as e:
        raise UnsafeExpressionError(f"unknown symbol: {name}") from e


def _eval_meson_node(node: ast.AST, symbols: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (str, int, float, bool)) or node.value is None:
            return node.value
        raise UnsafeExpressionError(f"unsupported literal: {node.value!r}")

    if isinstance(node, ast.Name):
        return _resolve_name(node.id, symbols)

    if isinstance(node, ast.List):
        return [_eval_meson_node(element, symbols) for element in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_eval_meson_node(element, symbols) for element in node.elts)

    if isinstance(node, ast.Set):
        return {_eval_meson_node(element, symbols) for element in node.elts}

    if isinstance(node, ast.UnaryOp):
        value = _eval_meson_node(node.operand, symbols)
        if not isinstance(value, (int, float)):
            raise UnsafeExpressionError("unary operators only supported for numbers")
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.USub):
            return -value
        raise UnsafeExpressionError("unsupported unary operator")

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _eval_meson_node(node.left, symbols)
        right = _eval_meson_node(node.right, symbols)

        if isinstance(left, list) and isinstance(right, list):
            return left + right
        if isinstance(left, tuple) and isinstance(right, tuple):
            return left + right
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left + right

        raise UnsafeExpressionError("unsupported '+' operands")

    raise UnsafeExpressionError(f"unsupported expression node: {type(node).__name__}")


def _eval_condition_node(node: ast.AST, symbols: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (str, int, float, bool)) or node.value is None:
            return node.value
        raise UnsafeExpressionError(f"unsupported literal: {node.value!r}")

    if isinstance(node, ast.Name):
        return _resolve_name(node.id, symbols)

    if isinstance(node, ast.Attribute):
        obj = _eval_condition_node(node.value, symbols)
        if node.attr.startswith("_"):
            raise UnsafeExpressionError("private attribute access is not allowed")
        try:
            return getattr(obj, node.attr)
        except AttributeError as e:
            raise UnsafeExpressionError(f"unknown attribute: {node.attr}") from e

    if isinstance(node, ast.List):
        return [_eval_condition_node(element, symbols) for element in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_eval_condition_node(element, symbols) for element in node.elts)

    if isinstance(node, ast.Set):
        return {_eval_condition_node(element, symbols) for element in node.elts}

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            for value_node in node.values:
                value = _eval_condition_node(value_node, symbols)
                if not value:
                    return False
            return True
        if isinstance(node.op, ast.Or):
            for value_node in node.values:
                value = _eval_condition_node(value_node, symbols)
                if value:
                    return True
            return False
        raise UnsafeExpressionError("unsupported boolean operator")

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval_condition_node(node.operand, symbols)

    if isinstance(node, ast.Compare):
        left = _eval_condition_node(node.left, symbols)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_condition_node(comparator, symbols)

            if isinstance(op, ast.Eq):
                matched = left == right
            elif isinstance(op, ast.NotEq):
                matched = left != right
            elif isinstance(op, ast.In):
                matched = left in right
            elif isinstance(op, ast.NotIn):
                matched = left not in right
            elif isinstance(op, ast.Is):
                matched = left is right
            elif isinstance(op, ast.IsNot):
                matched = left is not right
            elif isinstance(op, ast.Lt):
                matched = left < right
            elif isinstance(op, ast.LtE):
                matched = left <= right
            elif isinstance(op, ast.Gt):
                matched = left > right
            elif isinstance(op, ast.GtE):
                matched = left >= right
            else:
                raise UnsafeExpressionError(f"unsupported comparison operator: {type(op).__name__}")

            if not matched:
                return False
            left = right
        return True

    if isinstance(node, ast.Call):
        if node.keywords:
            raise UnsafeExpressionError("keyword arguments are not allowed")

        if not isinstance(node.func, ast.Attribute):
            raise UnsafeExpressionError("only method calls are allowed")

        target = _eval_condition_node(node.func.value, symbols)
        method_name = node.func.attr
        if method_name.startswith("_"):
            raise UnsafeExpressionError("private method calls are not allowed")

        if method_name not in {"startswith", "endswith"}:
            raise UnsafeExpressionError(f"unsupported method call: {method_name}")

        args = [_eval_condition_node(arg, symbols) for arg in node.args]
        method = getattr(target, method_name)
        return method(*args)

    raise UnsafeExpressionError(f"unsupported expression node: {type(node).__name__}")
