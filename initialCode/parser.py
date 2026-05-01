from __future__ import annotations

from dataclasses import dataclass

from ifp_ast import (
    CHARS,
    CHARS_DECODED,
    TBinOp,
    TBool,
    TIf,
    TInt,
    TLam,
    TString,
    TUnOp,
    TVar,
    Term,
)


@dataclass(frozen=True)
class ParseError(Exception):
    kind: str
    index: int | None = None
    ch: str | None = None

    def __str__(self) -> str:
        if self.kind == "UnexpectedChar":
            return f"UnexpectedChar({self.ch!r}, {self.index})"
        if self.kind == "UnusedInput":
            return f"UnusedInput({self.index})"
        return "UnexpectedEOF"


_DIGIT = {ch: i for i, ch in enumerate(CHARS)}
_DECODE = {src: dst for src, dst in zip(CHARS, CHARS_DECODED)}


def _from_base94(body: str) -> int:
    if body == "":
        raise ValueError("empty base-94 body")
    n = 0
    for ch in body:
        n = n * 94 + _DIGIT[ch]
    return n


def _decode_string(body: str) -> str:
    return "".join(_DECODE[ch] for ch in body)


def p_term(inp: str) -> Term:
    tokens = inp.split()
    pos = 0

    def parse_one() -> Term:
        nonlocal pos
        if pos >= len(tokens):
            raise ParseError("UnexpectedEOF")

        tok = tokens[pos]
        pos += 1
        if tok == "":
            raise ParseError("UnexpectedEOF")

        ind, body = tok[0], tok[1:]

        try:
            if ind == "T" and body == "":
                return TBool(True)
            if ind == "F" and body == "":
                return TBool(False)
            if ind == "I" and body != "":
                return TInt(_from_base94(body))
            if ind == "S":
                return TString(_decode_string(body))
            if ind == "U" and len(body) == 1:
                return TUnOp(body, parse_one())
            if ind == "B" and len(body) == 1:
                left = parse_one()
                right = parse_one()
                return TBinOp(left, body, right)
            if ind == "?" and body == "":
                cond = parse_one()
                true_branch = parse_one()
                false_branch = parse_one()
                return TIf(cond, true_branch, false_branch)
            if ind == "L" and body != "":
                return TLam(_from_base94(body), parse_one())
            if ind == "v" and body != "":
                return TVar(_from_base94(body))
        except KeyError as exc:
            raise ParseError("UnexpectedChar", None, str(exc)) from exc
        except ValueError as exc:
            raise ParseError("UnexpectedChar", None, str(exc)) from exc

        raise ParseError("UnexpectedChar", pos - 1, tok[0])

    result = parse_one()
    if pos != len(tokens):
        raise ParseError("UnusedInput", pos)
    return result
