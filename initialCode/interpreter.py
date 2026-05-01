from __future__ import annotations

from dataclasses import dataclass

from ifp_ast import TBinOp, TBool, TIf, TInt, TLam, TString, TUnOp, TVar, Term
from printer import encode_string, to_base94


MAX_STEPS = 10_000_000


class InterpreterError(Exception):
    pass


class BetaReductionLimit(InterpreterError):
    pass


class ScopeError(InterpreterError):
    pass


class TypeError_(InterpreterError):
    pass


class ArithmeticError_(InterpreterError):
    pass


class UnknownUnOp(InterpreterError):
    def __init__(self, op: str):
        super().__init__(f"Unknown unary operator: {op}")
        self.op = op


class UnknownBinOp(InterpreterError):
    def __init__(self, op: str):
        super().__init__(f"Unknown binary operator: {op}")
        self.op = op


@dataclass
class VInt:
    value: int


@dataclass
class VBool:
    value: bool


@dataclass
class VString:
    value: str


@dataclass
class VClosure:
    var: int
    body: Term
    env: dict[int, "Thunk"]


Value = VInt | VBool | VString | VClosure


@dataclass
class Thunk:
    kind: str
    value: Value | None = None
    steps: int = 0
    term: Term | None = None
    env: dict[int, "Thunk"] | None = None


def _to_term(v: Value) -> Term:
    if isinstance(v, VInt):
        return TInt(v.value)
    if isinstance(v, VBool):
        return TBool(v.value)
    if isinstance(v, VString):
        return TString(v.value)
    if isinstance(v, VClosure):
        return TLam(v.var, v.body)
    raise TypeError(f"Unknown value type: {type(v).__name__}")


def _expect_int(v: Value) -> int:
    if not isinstance(v, VInt):
        raise TypeError_("Expected integer")
    return v.value


def _expect_bool(v: Value) -> bool:
    if not isinstance(v, VBool):
        raise TypeError_("Expected boolean")
    return v.value


def _expect_string(v: Value) -> str:
    if not isinstance(v, VString):
        raise TypeError_("Expected string")
    return v.value


def _trunc_div(a: int, b: int) -> int:
    if b == 0:
        raise ArithmeticError_("Division by zero")
    return abs(a) // abs(b) * (-1 if (a < 0) ^ (b < 0) else 1)


def _trunc_mod(a: int, b: int) -> int:
    if b == 0:
        raise ArithmeticError_("Modulo by zero")
    return a - _trunc_div(a, b) * b


def interpret(check_max: bool, term: Term) -> tuple[Term, int]:
    steps = 0

    def bump_beta() -> None:
        nonlocal steps
        steps += 1
        if check_max and steps > MAX_STEPS:
            raise BetaReductionLimit(f"Beta reduction limit exceeded: {MAX_STEPS}")

    def force(th: Thunk) -> Value:
        if th.kind == "value":
            if th.value is None:
                raise InterpreterError("Invalid value thunk")
            return th.value
        if th.kind == "thunk":
            if th.term is None or th.env is None:
                raise InterpreterError("Invalid delayed thunk")
            # Call-by-name: do NOT memoize. Re-evaluate every occurrence independently.
            return eval_term(th.term, dict(th.env))
        raise InterpreterError(f"Unknown thunk kind: {th.kind}")

    def eval_term(t: Term, env: dict[int, Thunk]) -> Value:
        if isinstance(t, TInt):
            return VInt(t.value)
        if isinstance(t, TBool):
            return VBool(t.value)
        if isinstance(t, TString):
            return VString(t.value)
        if isinstance(t, TVar):
            if t.value not in env:
                raise ScopeError(f"Unbound variable: {t.value}")
            return force(env[t.value])
        if isinstance(t, TLam):
            return VClosure(t.var, t.body, dict(env))
        if isinstance(t, TIf):
            cond = _expect_bool(eval_term(t.cond, env))
            return eval_term(t.true_branch if cond else t.false_branch, env)
        if isinstance(t, TUnOp):
            v = eval_term(t.term, env)
            if t.op == "-":
                return VInt(-_expect_int(v))
            if t.op == "!":
                return VBool(not _expect_bool(v))
            if t.op == "#":
                s = _expect_string(v)
                enc = encode_string(s)
                n = 0
                for ch in enc:
                    n = n * 94 + (ord(ch) - 33)
                return VInt(n)
            if t.op == "$":
                n = _expect_int(v)
                body = to_base94(n)
                if body is None:
                    raise TypeError_("Cannot convert negative integer to string")
                return VString("".join(chr(ord_ch) for ord_ch in [])) if False else VString(_decode_base94_string_body(body))
            raise UnknownUnOp(t.op)
        if isinstance(t, TBinOp):
            if t.op == "$":
                fn = eval_term(t.left, env)
                if not isinstance(fn, VClosure):
                    raise TypeError_("Left side of function application must be a lambda")
                bump_beta()
                new_env = dict(fn.env)
                new_env[fn.var] = Thunk(kind="thunk", term=t.right, env=dict(env))
                return eval_term(fn.body, new_env)

            left = eval_term(t.left, env)
            right = eval_term(t.right, env)

            if t.op == "+":
                return VInt(_expect_int(left) + _expect_int(right))
            if t.op == "-":
                return VInt(_expect_int(left) - _expect_int(right))
            if t.op == "*":
                return VInt(_expect_int(left) * _expect_int(right))
            if t.op == "/":
                return VInt(_trunc_div(_expect_int(left), _expect_int(right)))
            if t.op == "%":
                return VInt(_trunc_mod(_expect_int(left), _expect_int(right)))
            if t.op == "<":
                return VBool(_expect_int(left) < _expect_int(right))
            if t.op == ">":
                return VBool(_expect_int(left) > _expect_int(right))
            if t.op == "=":
                if type(left) is not type(right):
                    return VBool(False)
                return VBool(left == right)
            if t.op == "|":
                return VBool(_expect_bool(left) or _expect_bool(right))
            if t.op == "&":
                return VBool(_expect_bool(left) and _expect_bool(right))
            if t.op == ".":
                return VString(_expect_string(left) + _expect_string(right))
            if t.op == "T":
                return VString(_expect_string(right)[: _expect_int(left)])
            if t.op == "D":
                return VString(_expect_string(right)[_expect_int(left) :])
            raise UnknownBinOp(t.op)
        raise TypeError(f"Unknown term type: {type(t).__name__}")

    def _decode_base94_string_body(body: str) -> str:
        from ifp_ast import CHARS, CHARS_DECODED
        decode = {src: dst for src, dst in zip(CHARS, CHARS_DECODED)}
        return "".join(decode[ch] for ch in body)

    result = eval_term(term, {})
    return _to_term(result), steps
