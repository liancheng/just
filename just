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
            "document": Document.from_tree,
            "id": Id.from_tree,
            "local_bind": Local.from_tree,
            "number": Number.from_tree,
            "string_content": Str.from_tree,
        }

        return dispatch[node.type](node)

    def to(self, expect_type: Type[ExprType]) -> ExprType:
        if not isinstance(self, expect_type):
            raise TypeError(
                f"Expected {expect_type.__name__}, got {type(self).__name__}"
            )

        return cast(ExprType, self)


@dataclass
class Document(Expr):
    body: Expr

    @staticmethod
    def from_tree(node: T.Node) -> "Document":
        return Document(
            range=lsp_range_of(node),
            body=Expr.from_tree(node.named_children[0]),
        )


@dataclass
class Id(Expr):
    name: str

    @staticmethod
    def from_tree(node: T.Node) -> "Id":
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
        assert node.text is not None
        return Str(
            range=lsp_range_of(node),
            value=node.text.decode(),
        )


@dataclass
class Bool(Expr):
    value: bool

    @staticmethod
    def from_tree(node: T.Node) -> "Bool":
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
        return Array(
            range=lsp_range_of(node),
            values=[Expr.from_tree(c) for c in node.named_children],
        )


@dataclass
class Bind:
    range: L.Range
    id: Id
    value: Expr

    @staticmethod
    def from_tree(node: T.Node) -> "Bind":
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

        while occurrence > 0:
            occurrence -= 1
            character = self.lines[line].find(needle, character + len(needle))

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
    def test_parse_number(self):
        t = TestDocument("1")

        self.assertEqual(
            t.document.body,
            Number(range=t.full_range, value=1),
        )

    def test_parse_local(self):
        t = TestDocument("local x = 1; x")

        self.assertEqual(
            t.document.body,
            Local(
                range=t.full_range,
                binds=[
                    Bind(
                        range=t.start_of("x").to(t.end_of("1")),
                        id=Id(t.range_of("x"), name="x"),
                        value=Number(t.start_of("1").length(1), value=1),
                    ),
                ],
                body=Id(t.range_of("x", occurrence=2), name="x"),
            ),
        )

    def test_parse_empty_array(self):
        t = TestDocument("[]")

        self.assertEqual(
            t.document.body,
            Array(
                range=t.full_range,
                values=[],
            ),
        )

    def test_parse_array(self):
        t = TestDocument("[1, 2]")

        self.assertEqual(
            t.document.body,
            Array(
                range=t.full_range,
                values=[
                    Number(range=t.range_of("1"), value=1),
                    Number(range=t.range_of("2"), value=2),
                ],
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
