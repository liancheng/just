#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "pygls",
#   "rich",
#   "tree-sitter",
#   "tree-sitter-language-pack",
#   "typer",
# ]
# ///

import logging
import unittest
from dataclasses import dataclass
from enum import StrEnum
from textwrap import dedent
from typing import Callable, Type, TypeVar, cast

import lsprotocol.types as L
import tree_sitter as T
import typer
from pygls.lsp.server import LanguageServer
from tree_sitter_language_pack import get_language, get_parser

app = typer.Typer(
    no_args_is_help=True,
    rich_markup_mode="markdown",
)

logging.basicConfig(
    filename="pygls.log",
    filemode="w",
    level=logging.DEBUG,
)

ExprType = TypeVar("ExprType")


@dataclass
class Expr:
    range: L.Range

    @staticmethod
    def from_tree(node: T.Node) -> "Expr":
        dispatch: dict[str, Callable[[T.Node], "Expr"]] = {
            "array": Array.from_tree,
            "binary": Binary.from_tree,
            "document": Document.from_tree,
            "false": Bool.from_tree,
            "forloop": ListComp.from_tree,
            "id": Id.from_tree,
            "local_bind": Local.from_tree,
            "number": Number.from_tree,
            "string": Str.from_tree,
            "true": Bool.from_tree,
        }

        return dispatch[node.type](node)

    def to(self, expect_type: Type[ExprType]) -> ExprType:
        if not isinstance(self, expect_type):
            raise TypeError(
                f"Expected {expect_type.__name__}, but got {type(self).__name__}"
            )

        return cast(ExprType, self)

    def bin_op(self, op: "Operator", rhs: "Expr") -> "Binary":
        range = L.Range(self.range.start, rhs.range.end)
        return Binary(range, op, self, rhs)

    def __add__(self, rhs: "Expr") -> "Binary":
        return self.bin_op(Operator.Plus, rhs)

    def __sub__(self, rhs: "Expr") -> "Binary":
        return self.bin_op(Operator.Minus, rhs)

    def __mul__(self, rhs: "Expr") -> "Binary":
        return self.bin_op(Operator.Multiply, rhs)

    def __truediv__(self, rhs: "Expr") -> "Binary":
        return self.bin_op(Operator.Divide, rhs)

    def __lt__(self, rhs: "Expr") -> "Binary":
        return self.bin_op(Operator.LT, rhs)

    def __le__(self, rhs: "Expr") -> "Binary":
        return self.bin_op(Operator.LE, rhs)

    def __gt__(self, rhs: "Expr") -> "Binary":
        return self.bin_op(Operator.GT, rhs)

    def __ge__(self, rhs: "Expr") -> "Binary":
        return self.bin_op(Operator.GE, rhs)

    def __eq__(self, rhs: "Expr") -> "Binary":
        return self.bin_op(Operator.Eq, rhs)

    def __ne__(self, rhs: "Expr") -> "Binary":
        return self.bin_op(Operator.NotEq, rhs)


@dataclass
class Document(Expr):
    body: Expr

    @staticmethod
    def from_tree(node: T.Node) -> "Document":
        assert node.type == "document"
        return Document(
            range=lsp_range_of(node),
            body=Expr.from_tree(node.named_children[0]),
        )


@dataclass
class Id(Expr):
    name: str

    @staticmethod
    def from_tree(node: T.Node) -> "Id":
        assert node.type == "id"
        assert node.text is not None

        return Id(
            range=lsp_range_of(node),
            name=node.text.decode(),
        )


@dataclass
class Number(Expr):
    value: float

    @staticmethod
    def from_tree(node: T.Node) -> "Number":
        assert node.type == "number"
        assert node.text is not None

        return Number(
            range=lsp_range_of(node),
            value=float(node.text.decode()),
        )


@dataclass
class Str(Expr):
    raw: str

    @staticmethod
    def from_tree(node: T.Node) -> "Str":
        assert node.type == "string"

        _, content, _ = node.named_children
        assert content.text is not None

        return Str(
            range=lsp_range_of(node),
            # The raw string content before escaping or indentations are handled.
            raw=content.text.decode(),
        )


@dataclass
class Bool(Expr):
    value: bool

    @staticmethod
    def from_tree(node: T.Node) -> "Bool":
        assert node.type in ["true", "false"]
        assert node.text is not None

        return Bool(
            range=lsp_range_of(node),
            value=node.text.decode() == "true",
        )


@dataclass
class Array(Expr):
    values: list[Expr]

    @staticmethod
    def from_tree(node: T.Node) -> "Array":
        assert node.type == "array"
        return Array(
            range=lsp_range_of(node),
            values=[Expr.from_tree(c) for c in node.named_children],
        )


class Operator(StrEnum):
    Multiply = "*"
    Divide = "/"
    Modulus = "%"

    Plus = "+"
    Minus = "-"

    ShiftLeft = "<<"
    ShiftRight = ">>"

    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    In = "in"

    Eq = "=="
    NotEq = "!="

    BitAnd = "&"
    BitXor = "^"
    BitOr = "|"
    And = "&&"
    Or = "||"


@dataclass
class Binary(Expr):
    op: Operator
    lhs: Expr
    rhs: Expr

    @staticmethod
    def from_tree(node: T.Node) -> "Binary":
        assert node.type == "binary"

        lhs, op, rhs = node.named_children
        assert op.text is not None

        return Binary(
            range=lsp_range_of(node),
            op=Operator(op.text.decode("utf-8")),
            lhs=Expr.from_tree(lhs),
            rhs=Expr.from_tree(rhs),
        )


@dataclass
class Bind:
    range: L.Range
    id: Id
    value: Expr

    @staticmethod
    def from_tree(node: T.Node) -> "Bind":
        assert node.type == "bind"
        [id, value] = node.named_children

        return Bind(
            range=L.Range(
                lsp_position_of(id.range.start_point),
                lsp_position_of(value.range.end_point),
            ),
            id=Expr.from_tree(id).to(Id),
            value=Expr.from_tree(value),
        )


@dataclass
class Local(Expr):
    binds: list[Bind]
    body: Expr

    @staticmethod
    def from_tree(node: T.Node) -> "Local":
        assert node.type == "local_bind"
        _, *binds, expr = node.named_children

        return Local(
            range=lsp_range_of(node),
            binds=[Bind.from_tree(b) for b in binds],
            body=Expr.from_tree(expr),
        )


@dataclass
class ForSpec:
    range: L.Range
    id: Id
    expr: Expr

    @staticmethod
    def from_tree(node: T.Node) -> "ForSpec":
        assert node.type == "forspec"
        id, expr = node.named_children

        return ForSpec(
            lsp_range_of(node),
            Id.from_tree(id),
            Expr.from_tree(expr),
        )


@dataclass
class IfSpec:
    range: L.Range
    condition: Expr

    @staticmethod
    def from_tree(node: T.Node) -> "IfSpec":
        assert node.type == "ifspec"
        [condition] = node.named_children

        return IfSpec(
            lsp_range_of(node),
            Expr.from_tree(condition),
        )


@dataclass
class ListComp(Expr):
    expr: Expr
    for_spec: ForSpec
    comp_spec: list[ForSpec | IfSpec]

    @staticmethod
    def from_tree(node: T.Node) -> "ListComp":
        if len(node.named_children) == 2:
            expr, for_spec = node.named_children
            comp_spec = []
        else:
            expr, for_spec, comp_spec = node.named_children
            comp_spec = comp_spec.named_children

        return ListComp(
            range=lsp_range_of(node),
            expr=Expr.from_tree(expr),
            for_spec=ForSpec.from_tree(for_spec),
            comp_spec=[
                ForSpec.from_tree(spec)
                if spec.type == "forspec"
                else IfSpec.from_tree(spec)
                for spec in comp_spec
            ],
        )


def lsp_position_of(point: T.Point) -> L.Position:
    return L.Position(point.row, point.column)


def lsp_range_of(node: T.Node) -> L.Range:
    return L.Range(
        lsp_position_of(node.range.start_point),
        lsp_position_of(node.range.end_point),
    )


server = LanguageServer("pj", "v0.1")


def parse(source: str) -> T.Node:
    return get_parser("jsonnet").parse(source.encode("utf-8")).root_node


def document_root(uri: str) -> T.Node:
    return parse(server.workspace.get_text_document(uri).source)


@server.feature(L.TEXT_DOCUMENT_COMPLETION)
def completions(params: L.CompletionParams):
    root_node = document_root(params.text_document.uri)
    query = T.Query(
        get_language("jsonnet"),
        dedent(
            """\
            (local_bind
              (bind (id) @local-bind.function
                    (params)))

            (local_bind
              (bind (id) @local-bind.variable
                    !params))

            (field
              (fieldname (id) @field.function)
              (params))

            (field
              (fieldname (id) @field)
              !params)
            """,
        ),
    )

    captures = T.QueryCursor(query).captures(root_node)

    def completion_items():
        patterns = {
            "local-bind.variable": L.CompletionItemKind.Variable,
            "local-bind.function": L.CompletionItemKind.Function,
            "field.function": L.CompletionItemKind.Method,
            "field": L.CompletionItemKind.Field,
        }

        for capture, kind in patterns.items():
            for node in captures.get(capture, []):
                if node.text is not None:
                    text = node.text.decode("utf-8")
                    yield L.CompletionItem(label=text, kind=kind)

    return L.CompletionList(
        is_incomplete=False,
        items=list(completion_items()),
    )


@dataclass
class ExprBuilder:
    range: L.Range

    def id(self, name: str) -> Id:
        return Id(self.range, name)

    def num(self, value: float) -> Number:
        return Number(self.range, value)

    def str(self, raw: str) -> Str:
        return Str(self.range, raw)

    def boolean(self, value: bool) -> Bool:
        return Bool(self.range, value)

    def arr(self, values: list[Expr]) -> Array:
        return Array(self.range, values)

    def bin_op(self, op: Operator | str, lhs: Expr, rhs: Expr) -> Binary:
        return Binary(self.range, Operator(op) if isinstance(op, str) else op, lhs, rhs)

    def bind(self, id: Id, value: Expr) -> Bind:
        return Bind(self.range, id, value)

    def local(self, binds: list[Bind], body: Expr) -> Local:
        return Local(self.range, binds, body)

    def list_comp(
        self,
        expr: Expr,
        specs: list[ForSpec | IfSpec] = [],
    ) -> ListComp:
        match specs:
            case [ForSpec() as for_spec, *comp_spec]:
                return ListComp(self.range, expr, for_spec, comp_spec)

            case []:
                raise RuntimeError(
                    "A list comprehension must have at least 1 for-spec."
                )

            case _:
                raise RuntimeError(
                    "The first spec of a list comprehension must be a for-spec."
                )

    def for_spec(self, id: Id, expr: Expr) -> ForSpec:
        return ForSpec(self.range, id, expr)

    def if_spec(self, condition: Expr) -> IfSpec:
        return IfSpec(self.range, condition)


def at(range: L.Range) -> ExprBuilder:
    return ExprBuilder(range)


class TestDocument:
    def __init__(self, source: str) -> None:
        self.lines = source.splitlines(keepends=False)
        self.root_node = parse(source)
        self.document = Document.from_tree(self.root_node)
        self.body_range = self.document.body.range

    def start_of(self, needle: str, line=1, occurrence=1) -> L.Position:
        assert line >= 1 and occurrence >= 1

        line -= 1
        occurrence -= 1

        character = self.lines[line].find(needle)
        assert character >= 0

        while occurrence > 0:
            occurrence -= 1
            character = self.lines[line].find(needle, character + len(needle))
            assert character >= 0

        return L.Position(line, character)

    def end_of(self, needle: str, line=1, occurrence=1) -> L.Position:
        pos = self.start_of(needle, line, occurrence)
        pos.character += len(needle)
        return pos

    def range_of(self, needle: str, line=1, occurrence=1) -> L.Range:
        start = self.start_of(needle, line, occurrence)
        end = L.Position(start.line, start.character + len(needle))
        return L.Range(start, end)

    def id(self, name: str, line=1, occurrence=1) -> Id:
        return at(self.range_of(name, line, occurrence)).id(name)


class TestAST(unittest.TestCase):
    def test_number(self):
        t = TestDocument("1")

        self.assertEqual(
            t.document.body,
            at(t.body_range).num(1),
        )

    def test_string(self):
        for input, expected in [
            ("'hello\\nworld'", "hello\\nworld"),
            ('"hello\\nworld"', "hello\\nworld"),
            ("@'hello\\nworld'", "hello\\nworld"),
            ('@"hello\\nworld"', "hello\\nworld"),
            ("|||\n  hello\n|||", "\n  hello\n"),
        ]:
            t = TestDocument(input)
            self.assertEqual(
                t.document.body,
                at(t.body_range).str(expected),
            )

    def test_local(self):
        t = TestDocument("local x = 1; x")

        self.assertEqual(
            t.document.body,
            at(t.body_range).local(
                binds=[
                    at(t.range_of("x = 1")).bind(
                        t.id("x"),
                        at(t.range_of("1")).num(1),
                    ),
                ],
                body=t.id("x", occurrence=2),
            ),
        )

    def test_local_multi_binds(self):
        t = TestDocument(
            dedent(
                """\
                local x = 1, y = 2;
                x + y
                """
            )
        )

        self.assertEqual(
            t.document.body,
            at(t.body_range).local(
                binds=[
                    at(t.range_of("x = 1")).bind(
                        t.id("x"),
                        at(t.range_of("1")).num(1),
                    ),
                    at(t.range_of("y = 2")).bind(
                        t.id("y"),
                        at(t.range_of("2")).num(2),
                    ),
                ],
                body=t.id("x", line=2) + t.id("y", line=2),
            ),
        )

    def test_empty_array(self):
        t = TestDocument("[]")

        self.assertEqual(
            t.document.body,
            at(t.body_range).arr([]),
        )

    def test_array(self):
        t = TestDocument("[1, true, '3']")

        self.assertEqual(
            t.document.body,
            at(t.body_range).arr(
                [
                    at(t.range_of("1")).num(1),
                    at(t.range_of("true")).boolean(True),
                    at(t.range_of("'3'")).str("3"),
                ],
            ),
        )

    def test_binary(self):
        for op in Operator:
            t = TestDocument(f"a {op.value} b")

            self.assertEqual(
                t.document.body,
                t.id("a").bin_op(op, t.id("b")),
            )

    def test_binary_precedences(self):
        t = TestDocument("a + b * c")

        self.assertEqual(
            t.document.body,
            t.id("a") + t.id("b") * t.id("c"),
        )

    def test_list_comp(self):
        t = TestDocument("[x for x in [1, 2] if x > 1]")

        self.assertEqual(
            t.document.body,
            at(t.body_range).list_comp(
                expr=t.id("x"),
                specs=[
                    at(t.range_of("for x in [1, 2]")).for_spec(
                        id=t.id("x", occurrence=2),
                        expr=at(t.range_of("[1, 2]")).arr(
                            [
                                at(t.range_of("1")).num(1.0),
                                at(t.range_of("2")).num(2.0),
                            ]
                        ),
                    ),
                    at(t.range_of("if x > 1")).if_spec(
                        t.id("x", occurrence=3)
                        > at(t.range_of("1", occurrence=2)).num(1.0),
                    ),
                ],
            ),
        )


@app.command()
def serve():
    server.start_io()


@app.command()
def test():
    unittest.main(argv=["--verbose", "--locals"])


if __name__ == "__main__":
    app()

# vim:ft=python tw=88
