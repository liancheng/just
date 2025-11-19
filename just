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
import sys
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
    value: str

    @staticmethod
    def from_tree(node: T.Node) -> "Str":
        assert node.type == "string"

        _, content, _ = node.named_children
        assert content.text is not None

        return Str(
            range=lsp_range_of(node),
            value=content.text.decode(),
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


class PositionDSL(L.Position):
    def length(self, n: int) -> L.Range:
        return L.Range(self, L.Position(self.line, self.character + n))

    def to(self, end: L.Position) -> L.Range:
        return L.Range(self, end)


class TestDocument:
    def __init__(self, source: str) -> None:
        self.lines = source.splitlines(keepends=False)
        self.root_node = parse(source)
        self.document = Document.from_tree(self.root_node)
        self.full_range = lsp_range_of(self.root_node)

    def start_of(self, needle: str, line=1, occurrence=1) -> PositionDSL:
        assert line >= 1 and occurrence >= 1

        line -= 1
        occurrence -= 1

        character = self.lines[line].find(needle)
        assert character >= 0

        while occurrence > 0:
            occurrence -= 1
            character = self.lines[line].find(needle, character + len(needle))
            assert character >= 0

        return PositionDSL(line, character)

    def end_of(self, needle: str, line=1, occurrence=1) -> PositionDSL:
        pos = self.start_of(needle, line, occurrence)
        pos.character += len(needle)
        return pos

    def range_of(self, needle: str, line=1, occurrence=1) -> L.Range:
        return L.Range(
            self.start_of(needle, line, occurrence),
            self.end_of(needle, line, occurrence),
        )


def start_at(line: int, character: int) -> PositionDSL:
    return PositionDSL(line, character)


class TestAST(unittest.TestCase):
    def test_number(self):
        t = TestDocument("1")

        self.assertEqual(
            t.document.body,
            Number(range=t.full_range, value=1),
        )

    def test_single_quoted_string(self):
        t = TestDocument("'hello'")

        self.assertEqual(
            t.document.body,
            Str(range=t.full_range, value="hello"),
        )

    def test_double_quoted_string(self):
        t = TestDocument('"hello"')

        self.assertEqual(
            t.document.body,
            Str(range=t.full_range, value="hello"),
        )

    def test_verbatim_string(self):
        t = TestDocument(
            dedent(
                """\
                |||
                    hello
                |||
                """
            )
        )

        # Seems that the Jsonnet tree-sitter grammar does NOT handle indentations.
        self.assertEqual(
            t.document.body,
            Str(
                range=t.start_of("|||").to(t.end_of("|||", line=3)),
                value="\n    hello\n",
            ),
        )

    def test_local(self):
        t = TestDocument("local x = 1; x")

        self.assertEqual(
            t.document.body,
            Local(
                range=t.full_range,
                binds=[
                    Bind(
                        range=t.start_of("x").to(t.end_of("1")),
                        id=Id(t.range_of("x"), name="x"),
                        value=Number(t.range_of("1"), value=1),
                    ),
                ],
                body=Id(t.range_of("x", occurrence=2), name="x"),
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

        local = t.document.body.to(Local)

        self.assertEqual(
            local.binds,
            [
                Bind(
                    range=t.start_of("x").to(t.end_of("1")),
                    id=Id(range=t.range_of("x"), name="x"),
                    value=Number(range=t.range_of("1"), value=1.0),
                ),
                Bind(
                    range=t.start_of("y").to(t.end_of("2")),
                    id=Id(range=t.range_of("y"), name="y"),
                    value=Number(range=t.range_of("2"), value=2.0),
                ),
            ],
        )

        self.assertEqual(
            local.body,
            Binary(
                range=t.range_of("x + y", line=2),
                op=Operator.Plus,
                lhs=Id(range=t.range_of("x", line=2), name="x"),
                rhs=Id(range=t.range_of("y", line=2), name="y"),
            ),
        )

    def test_empty_array(self):
        t = TestDocument("[]")

        self.assertEqual(
            t.document.body,
            Array(
                range=t.full_range,
                values=[],
            ),
        )

    def test_array(self):
        t = TestDocument("[1, true, '3']")

        self.assertEqual(
            t.document.body,
            Array(
                range=t.full_range,
                values=[
                    Number(range=t.range_of("1"), value=1.0),
                    Bool(range=t.range_of("true"), value=True),
                    Str(range=t.range_of("'3'"), value="3"),
                ],
            ),
        )

    def test_binary(self):
        for op in Operator:
            t = TestDocument(f"a {op.value} b")

            self.assertEqual(
                t.document.body,
                Binary(
                    range=t.full_range,
                    op=op,
                    lhs=Id(range=t.range_of("a"), name="a"),
                    rhs=Id(range=t.range_of("b"), name="b"),
                ),
            )

    def test_binary_precedences(self):
        t = TestDocument("a + b * c")

        self.assertEqual(
            t.document.body,
            Binary(
                range=t.full_range,
                op=Operator.Plus,
                lhs=Id(range=t.range_of("a"), name="a"),
                rhs=Binary(
                    range=t.range_of("b * c"),
                    op=Operator.Multiply,
                    lhs=Id(range=t.range_of("b"), name="b"),
                    rhs=Id(range=t.range_of("c"), name="c"),
                ),
            ),
        )


@app.command()
def serve():
    server.start_io()


@app.command()
def test():
    unittest.main(argv=sys.argv[1:])


if __name__ == "__main__":
    app()

# vim:ft=python tw=88
