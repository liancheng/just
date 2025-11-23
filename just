#!/usr/bin/env -S uv run --script --offline
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
from contextlib import contextmanager
from dataclasses import dataclass, fields
from enum import StrEnum
from textwrap import dedent
from typing import Any, Type, TypeVar, cast

import lsprotocol.types as L
import tree_sitter as T
import typer
from pygls.lsp.server import LanguageServer
from tree_sitter_language_pack import get_parser

app = typer.Typer(
    no_args_is_help=True,
    rich_markup_mode="markdown",
)

logging.basicConfig(
    filename="pygls.log",
    filemode="w",
    level=logging.DEBUG,
)

log = logging.root


ExprType = TypeVar("ExprType")


@dataclass
class AST:
    range: L.Range

    @property
    def pretty_tree(self) -> str:
        @dataclass
        class PrettyNode:
            value: Any
            label: str | None = None

            def __repr__(self):
                match self.value:
                    case AST() as ast:
                        repr = f"{ast.__class__.__name__} [{ast.range}]"
                    case [_, *_]:
                        repr = "[...]"
                    case _:
                        repr = str(self.value)

                return repr if self.label is None else f"{self.label}={repr}"

            @property
            def children(self) -> list["PrettyNode"]:
                match self.value:
                    case AST() as ast:
                        return [
                            PrettyNode(getattr(ast, f.name), f.name)
                            for f in fields(ast)
                            if f.name != "range"
                        ]
                    case list() as elems:
                        return [PrettyNode(e) for e in elems]
                    case _:
                        return []

        def dump(buf: list[str], nodes: list[PrettyNode], branches: str = ""):
            for i in range(len(nodes)):
                last = i == len(nodes) - 1
                new_branch = " " * 4 if last else ":   "
                fork = "`-- " if last else ":-- "

                buf.append(f"{branches}{fork}{nodes[i]}")
                dump(buf, nodes[i].children, branches + new_branch)

        root = PrettyNode(self)
        lines = [str(root)]
        dump(lines, root.children)

        return "\n".join(lines)


@dataclass
class Expr(AST):
    @staticmethod
    def from_tree(node: T.Node) -> "Expr":
        global dispatch

        dispatch = {
            "anonymous_function": Fn.from_tree,
            "array": Array.from_tree,
            "assert": AssertExpr.from_tree,
            "binary": Binary.from_tree,
            "document": Document.from_tree,
            "false": Bool.from_tree,
            "forloop": ListComp.from_tree,
            "functioncall": Call.from_tree,
            "id": Id.from_tree,
            "import": Import.from_tree,
            "importstr": Import.from_tree,
            "local_bind": Local.from_tree,
            "number": Num.from_tree,
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
        return Binary(merge_ranges(self, rhs), op, self, rhs)

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

    def eq(self, rhs: object) -> "Binary":
        assert isinstance(rhs, Expr)
        return self.bin_op(Operator.Eq, rhs)

    def not_eq(self, rhs: "Expr") -> "Binary":
        return self.bin_op(Operator.NotEq, rhs)


@dataclass
class Document(Expr):
    body: Expr

    @staticmethod
    def from_tree(node: T.Node) -> "Document":
        assert node.type == "document"
        body, *_ = node.named_children
        return Document(range_of(node), Expr.from_tree(body))


@dataclass
class Id(Expr):
    name: str

    @staticmethod
    def from_tree(node: T.Node) -> "Id":
        assert node.type == "id"
        assert node.text is not None

        return Id(
            range=range_of(node),
            name=node.text.decode(),
        )

    def bind(self, value: Expr) -> "Bind":
        return Bind(merge_ranges(self, value), self, value)

    def arg(self, value: Expr) -> "Arg":
        return Arg(merge_ranges(self, value), value, self)


@dataclass
class Num(Expr):
    value: float

    @staticmethod
    def from_tree(node: T.Node) -> "Num":
        assert node.type == "number"
        assert node.text is not None

        return Num(
            range=range_of(node),
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
            range=range_of(node),
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
            range=range_of(node),
            value=node.text.decode() == "true",
        )


@dataclass
class Array(Expr):
    values: list[Expr]

    @staticmethod
    def from_tree(node: T.Node) -> "Array":
        assert node.type == "array"
        return Array(
            range=range_of(node),
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
            range=range_of(node),
            op=Operator(op.text.decode("utf-8")),
            lhs=Expr.from_tree(lhs),
            rhs=Expr.from_tree(rhs),
        )


# Unfortunately, `tree-sitter-jsonnet` did a bad job in parsing assert expressions
# (an expression following an assertion). Ideally, an `AssertExpr` should map to a
# single tree-sitter node containing an assertion followed by an expression. E.g., the
# given the following expression:
#
#   assert true: "never"; x
#
# it should be parsed into a single `<assert_expr>` node:
#
#   <assert_expr>     | assert true: "never"; x
#       <assert>      | assert true: "never"
#           <_expr>   |        true
#           <string>  |              "never"
#       <id>          |                       x
#
# However, `tree-sitter-jsonnet` explicitly hides the top node by prefixing the rule
# with an underscore (`_assert_expr`), and therefore the parser only returns the two
# child nodes:
#
#   <assert>          | assert true: "never"
#       <_expr>       |        true
#       <string>      |              "never"
#   <id>              |                       x
#
# This complicates parsing values in local bindings and function bodies, because a bound
# value and a function body can be either one node for a regular expression or two nodes
# for an assertion expression.
def parse_expr(nodes: list[T.Node]) -> Expr:
    assert 1 <= len(nodes) <= 2

    if len(nodes) == 2:
        value, _ = nodes
        assert value.type == "assert"
    else:
        [value] = nodes

    return Expr.from_tree(value)


@dataclass
class Bind(AST):
    id: Id
    value: Expr

    @staticmethod
    def from_tree(node: T.Node) -> "Bind":
        assert node.type == "bind"
        assert 2 <= len(node.named_children) <= 3

        if len(node.children_by_field_name("function")) == 0:
            id, *value_nodes = node.named_children
            return Bind(
                range=range_of(node),
                id=Id.from_tree(id),
                value=parse_expr(value_nodes),
            )
        else:
            # Following `id`, here could be:
            #
            #  - 0 or 1 `params` node holding the function parameter list:
            #    - 0 nodes if the parameter list is empty, or
            #    - 1 `params` node with all the parameters as its children.
            #  - 1 or 2 `body` nodes:
            #    - 2 nodes if the body is an assertion expression, or
            #    - 1 node for other regular expressions.
            id, *_ = node.named_children

            maybe_params = node.children_by_field_name("params")
            assert len(maybe_params) <= 1

            body_nodes = node.children_by_field_name("body")

            fn = Fn(
                range=range_of(node),
                params=[
                    Param.from_tree(param)
                    for params in maybe_params
                    for param in params.named_children
                ],
                body=parse_expr(body_nodes),
            )

            return Bind(
                range=fn.range,
                id=Id.from_tree(id),
                value=fn,
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
            range=range_of(node),
            binds=[Bind.from_tree(bind) for bind in binds],
            body=Expr.from_tree(expr),
        )


@dataclass
class Param(AST):
    name: Id
    value: Expr | None = None

    @staticmethod
    def from_tree(node: T.Node) -> "Param":
        assert node.type == "param"
        assert 1 <= len(node.named_children) <= 2

        id_node, *maybe_value = node.named_children

        return Param(
            range=range_of(node),
            name=Id.from_tree(id_node),
            value=next(iter(map(Expr.from_tree, maybe_value)), None),
        )


@dataclass
class Fn(Expr):
    params: list[Param]
    body: Expr

    @staticmethod
    def from_tree(node: T.Node) -> "Fn":
        assert node.type == "anonymous_function"
        assert 1 <= len(node.named_children) <= 2

        *maybe_params, body_node = node.named_children

        return Fn(
            range=range_of(node),
            params=[
                Param.from_tree(param)
                for params in maybe_params
                for param in params.named_children
            ],
            body=Expr.from_tree(body_node),
        )


@dataclass
class Arg(AST):
    value: Expr
    name: Id | None = None

    @staticmethod
    def from_tree(node: T.Node) -> "Arg":
        if node.type == "named_argument":
            assert len(node.named_children) == 2
            name, value = node.named_children

            return Arg(
                range=range_of(node),
                value=Expr.from_tree(value),
                name=Id.from_tree(name),
            )
        else:
            return Arg(
                range=range_of(node),
                value=Expr.from_tree(node),
            )


@dataclass
class Call(Expr):
    fn: Expr
    args: list[Arg]

    @staticmethod
    def from_tree(node: T.Node) -> "Call":
        assert node.type == "functioncall"
        assert len(node.named_children) >= 1

        fn, *maybe_args = node.named_children

        return Call(
            range=range_of(node),
            fn=Expr.from_tree(fn),
            args=[
                Arg.from_tree(arg) for args in maybe_args for arg in args.named_children
            ],
        )


@dataclass
class ForSpec(AST):
    id: Id
    expr: Expr

    @staticmethod
    def from_tree(node: T.Node) -> "ForSpec":
        assert node.type == "forspec"
        assert len(node.named_children) == 2

        id, expr = node.named_children

        return ForSpec(
            range_of(node),
            Id.from_tree(id),
            Expr.from_tree(expr),
        )


@dataclass
class IfSpec(AST):
    condition: Expr

    @staticmethod
    def from_tree(node: T.Node) -> "IfSpec":
        assert node.type == "ifspec"
        assert len(node.named_children) == 1

        return IfSpec(
            range_of(node),
            condition=next(map(Expr.from_tree, node.named_children)),
        )


@dataclass
class ListComp(Expr):
    expr: Expr
    for_spec: ForSpec
    comp_spec: list[ForSpec | IfSpec]

    @staticmethod
    def from_tree(node: T.Node) -> "ListComp":
        assert node.type == "forloop"
        assert len(node.named_children) >= 2

        expr, for_spec, *maybe_comp_spec = node.named_children

        return ListComp(
            range=range_of(node),
            expr=Expr.from_tree(expr),
            for_spec=ForSpec.from_tree(for_spec),
            comp_spec=[
                ForSpec.from_tree(spec)
                if spec.type == "forspec"
                else IfSpec.from_tree(spec)
                for comp_spec in maybe_comp_spec
                for spec in comp_spec.named_children
            ],
        )


@dataclass
class Import(Expr):
    type: str
    path: Str

    @staticmethod
    def from_tree(node: T.Node) -> "Import":
        assert node.type in ["import", "importstr"]
        assert len(node.named_children) == 1

        [path] = node.named_children
        return Import(
            range_of(node),
            node.type,
            Str.from_tree(path),
        )


@dataclass
class Assert(AST):
    condition: Expr
    message: Expr | None = None

    @staticmethod
    def from_tree(node: T.Node) -> "Assert":
        assert node.type == "assert"
        assert 1 <= len(node.named_children) <= 2

        condition, *maybe_message = node.named_children
        return Assert(
            range_of(node),
            Expr.from_tree(condition),
            next(map(Expr.from_tree, maybe_message), None),
        )


@dataclass
class AssertExpr(Expr):
    assertion: Assert
    body: Expr

    @staticmethod
    def from_tree(node: T.Node) -> "Expr":
        body = node.next_named_sibling
        assert body is not None
        return AssertExpr(
            merge_ranges(node, body),
            Assert.from_tree(node),
            Expr.from_tree(body),
        )


class Visitor:
    def visit(self, tree: Expr):
        match tree:
            case Document() as e:
                self.visit_document(e)
            case Id() as e:
                self.visit_id(e)
            case Str() as e:
                self.visit_str(e)
            case Num() as e:
                self.visit_num(e)
            case Bool() as e:
                self.visit_bool(e)
            case Array() as e:
                self.visit_array(e)
            case Binary() as e:
                self.visit_binary(e)
            case Local() as e:
                self.visit_local(e)
            case Fn() as e:
                self.visit_fn(e)
            case Call() as e:
                self.visit_call(e)
            case ListComp() as e:
                self.visit_list_comp(e)
            case Import() as e:
                self.visit_import(e)
            case AssertExpr() as e:
                self.visit_assert_expr(e)

    def visit_document(self, e: Document):
        self.visit(e.body)

    def visit_id(self, e: Id):
        del e

    def visit_str(self, e: Str):
        del e

    def visit_num(self, e: Num):
        del e

    def visit_bool(self, e: Bool):
        del e

    def visit_array(self, e: Array):
        for v in e.values:
            self.visit(v)

    def visit_binary(self, e: Binary):
        self.visit(e.lhs)
        self.visit(e.rhs)

    def visit_local(self, e: Local):
        for b in e.binds:
            self.visit_bind(b)
        self.visit(e.body)

    def visit_bind(self, b: Bind):
        self.visit_id(b.id)
        self.visit(b.value)

    def visit_fn(self, e: Fn):
        for p in e.params:
            self.visit_param(p)
        self.visit(e.body)

    def visit_param(self, p: Param):
        self.visit_id(p.name)
        if p.value is not None:
            self.visit(p.value)

    def visit_call(self, e: Call):
        self.visit(e.fn)
        for a in e.args:
            self.visit_arg(a)

    def visit_arg(self, a: Arg):
        if a.name is not None:
            self.visit(a.name)
        self.visit(a.value)

    def visit_list_comp(self, e: ListComp):
        self.visit_for_spec(e.for_spec)

        for s in e.comp_spec:
            match s:
                case ForSpec() as f:
                    self.visit_for_spec(f)
                case IfSpec() as i:
                    self.visit_if_spec(i)

        self.visit(e.expr)

    def visit_for_spec(self, s: ForSpec):
        self.visit(s.expr)
        self.visit_id(s.id)

    def visit_if_spec(self, s: IfSpec):
        self.visit(s.condition)

    def visit_import(self, e: Import):
        self.visit_str(e.path)

    def visit_assert_expr(self, e: AssertExpr):
        self.visit_assert(e.assertion)
        self.visit(e.body)

    def visit_assert(self, a: Assert):
        self.visit(a.condition)
        if a.message is not None:
            self.visit(a.message)


def position_of(point: T.Point) -> L.Position:
    return L.Position(point.row, point.column)


def range_of(node: T.Node) -> L.Range:
    return L.Range(
        position_of(node.range.start_point),
        position_of(node.range.end_point),
    )


PositionLike = L.Position | T.Point


def position_before(lhs: PositionLike, rhs: PositionLike):
    match lhs:
        case L.Position() as p:
            lhs_line = p.line
            lhs_character = p.character
        case T.Point() as p:
            lhs_line = p.row
            lhs_character = p.column

    match rhs:
        case L.Position() as p:
            rhs_line = p.line
            rhs_character = p.character
        case T.Point() as p:
            rhs_line = p.row
            rhs_character = p.column

    return (
        lhs_line < rhs_line or lhs_line == rhs_line and lhs_character <= rhs_character
    )


RangeLike = L.Range | T.Range | T.Node | Expr


def merge_ranges(lhs: RangeLike, rhs: RangeLike) -> L.Range:
    match lhs:
        case L.Range() as r:
            start = r.start
        case T.Range() as r:
            start = position_of(r.start_point)
        case T.Node() as n:
            start = position_of(n.start_point)
        case Expr() as e:
            start = e.range.start

    match rhs:
        case L.Range() as r:
            end = r.end
        case T.Range() as r:
            end = position_of(r.end_point)
        case T.Node() as n:
            end = position_of(n.end_point)
        case Expr() as e:
            end = e.range.end

    assert position_before(start, end)

    return L.Range(start, end)


def parse(source: str) -> T.Node:
    return get_parser("jsonnet").parse(source.encode("utf-8")).root_node


class Indexer(Visitor):
    def __init__(self, tree: Expr) -> None:
        self.root_symbol = L.DocumentSymbol(
            name="__root__",
            kind=L.SymbolKind.Variable,
            range=tree.range,
            selection_range=tree.range,
        )

        self.breadcrumbs = [self.root_symbol]
        self.visit(tree)

    def visit_bind(self, b: Bind):
        var_symbol = L.DocumentSymbol(
            name=b.id.name,
            kind=L.SymbolKind.Variable,
            range=b.id.range,
            selection_range=b.range,
        )

        with self.parent_symbol(var_symbol):
            self.visit(b.value)

    @contextmanager
    def parent_symbol(self, symbol: L.DocumentSymbol):
        parent = self.breadcrumbs[-1]

        match parent.children:
            case list():
                parent.children.append(symbol)
            case None:
                parent.children = [symbol]

        self.breadcrumbs.append(symbol)

        try:
            yield symbol
        finally:
            self.breadcrumbs.pop()


server = LanguageServer("just", "v0.1")


def document_root(uri: str) -> T.Node:
    return parse(server.workspace.get_text_document(uri).source)


@server.feature(L.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
def document_symbol(params: L.DocumentColorParams):
    root_node = document_root(params.text_document.uri)
    tree = Expr.from_tree(root_node)
    indexer = Indexer(tree)
    return indexer.root_symbol.children


class TestDocument:
    def __init__(self, source: str) -> None:
        self.lines = source.splitlines(keepends=False)
        self.body = Document.from_tree(parse(source)).body

    def start_of(self, needle: str, line=1, nth=1) -> L.Position:
        assert line >= 1 and nth >= 1

        line -= 1
        nth -= 1

        character = self.lines[line].find(needle)
        assert character >= 0

        while nth > 0:
            nth -= 1
            character = self.lines[line].find(needle, character + len(needle))
            assert character >= 0

        return L.Position(line, character)

    def end_of(self, needle: str, line=1, nth=1) -> L.Position:
        pos = self.start_of(needle, line, nth)
        pos.character += len(needle)
        return pos

    def range_of(self, needle: str, line=1, nth=1) -> L.Range:
        start = self.start_of(needle, line, nth)
        end = L.Position(start.line, start.character + len(needle))
        return L.Range(start, end)

    def id(self, name: str, line=1, nth=1) -> Id:
        return Id(self.range_of(name, line, nth), name)

    def boolean(self, value: bool, line=1, nth=1) -> Bool:
        needle = "true" if value else "false"
        range = self.range_of(needle, line, nth)
        return Bool(range, value)

    def num(self, value: float | int, literal: str | None = None, line=1, nth=1) -> Num:
        match value, literal:
            case int(), None:
                literal = str(value)
                value = float(value)
            case _:
                assert literal is not None

        return Num(self.range_of(literal, line, nth), value)

    def str(self, value: str, literal: str, line=1, nth=1) -> Str:
        return Str(self.range_of(literal, line, nth), value)

    def param(
        self, name: str, line: int = 1, nth: int = 1, value: Expr | None = None
    ) -> Param:
        id = self.id(name, line, nth)
        range = id.range if value is None else merge_ranges(id, value)
        return Param(range, id, value)

    def arg(self, value: Expr) -> Arg:
        return Arg(value.range, value)


class TestAST(unittest.TestCase):
    def assertExprEqual(self, actual, expected):
        self.assertMultiLineEqual(
            actual.pretty_tree,
            expected.pretty_tree,
        )

    def test_pretty_tree(self):
        def check(source: str, expected: str):
            self.assertMultiLineEqual(
                TestDocument(source).body.pretty_tree,
                expected.strip(),
            )

        check(
            "1",
            dedent(
                """\
                Num [0:0-0:1]
                `-- value=1.0
                """
            ),
        )

        check(
            "'f' + 1",
            dedent(
                """\
                Binary [0:0-0:7]
                :-- op=+
                :-- lhs=Str [0:0-0:3]
                :   `-- raw=f
                `-- rhs=Num [0:6-0:7]
                    `-- value=1.0
                """
            ),
        )

        check(
            "local x = 1; x + 2",
            dedent(
                """\
                Local [0:0-0:18]
                :-- binds=[...]
                :   `-- Bind [0:6-0:11]
                :       :-- id=Id [0:6-0:7]
                :       :   `-- name=x
                :       `-- value=Num [0:10-0:11]
                :           `-- value=1.0
                `-- body=Binary [0:13-0:18]
                    :-- op=+
                    :-- lhs=Id [0:13-0:14]
                    :   `-- name=x
                    `-- rhs=Num [0:17-0:18]
                        `-- value=2.0
                """
            ),
        )

    def test_number(self):
        t = TestDocument("1")

        self.assertExprEqual(
            t.body,
            t.num(1),
        )

    def test_string(self):
        for literal, expected in [
            ("'hello\\nworld'", "hello\\nworld"),
            ('"hello\\nworld"', "hello\\nworld"),
            ("@'hello\\nworld'", "hello\\nworld"),
            ('@"hello\\nworld"', "hello\\nworld"),
            ("|||\n  hello\n|||", "\n  hello\n"),
        ]:
            t = TestDocument(literal)
            self.assertExprEqual(
                t.body,
                Str(t.body.range, expected),
            )

    def test_local(self):
        t = TestDocument("local x = 1; x")

        self.assertExprEqual(
            t.body,
            Local(
                range=t.body.range,
                binds=[t.id("x").bind(t.num(1))],
                body=t.id("x", nth=2),
            ),
        )

    def test_local_bind_fn(self):
        t = TestDocument(
            dedent(
                """\
                local
                    f1(x) = x + 1,
                    f2(y, z) = y + z;
                f2(f1(3), z = 4)
                """
            )
        )

        self.assertExprEqual(
            t.body,
            Local(
                range=t.body.range,
                binds=[
                    t.id("f1", line=2).bind(
                        Fn(
                            range=t.range_of("f1(x) = x + 1", line=2),
                            params=[t.param("x", line=2)],
                            body=t.id("x", line=2, nth=2) + t.num(1, line=2, nth=2),
                        )
                    ),
                    t.id("f2", line=3).bind(
                        Fn(
                            range=t.range_of("f2(y, z) = y + z", line=3),
                            params=[
                                t.param("y", line=3),
                                t.param("z", line=3),
                            ],
                            body=t.id("y", line=3, nth=2) + t.id("z", line=3, nth=2),
                        )
                    ),
                ],
                body=Call(
                    range=t.range_of("f2(f1(3), z = 4)", line=4),
                    fn=t.id("f2", line=4),
                    args=[
                        t.arg(
                            Call(
                                range=t.range_of("f1(3)", line=4),
                                fn=t.id("f1", line=4),
                                args=[t.arg(t.num(3, line=4))],
                            )
                        ),
                        t.id("z", line=4).arg(t.num(4, line=4)),
                    ],
                ),
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

        self.assertExprEqual(
            t.body,
            Local(
                range=t.body.range,
                binds=[
                    t.id("x").bind(t.num(1)),
                    t.id("y").bind(t.num(2)),
                ],
                body=t.id("x", line=2) + t.id("y", line=2),
            ),
        )

    def test_empty_array(self):
        t = TestDocument("[]")

        self.assertExprEqual(
            t.body,
            Array(range=t.body.range, values=[]),
        )

    def test_array(self):
        t = TestDocument("[1, true, '3']")

        self.assertExprEqual(
            t.body,
            Array(
                range=t.body.range,
                values=[
                    t.num(1),
                    t.boolean(True),
                    t.str("3", literal="'3'"),
                ],
            ),
        )

    def test_binary(self):
        for op in Operator:
            t = TestDocument(f"a {op.value} b")

            self.assertExprEqual(
                t.body,
                t.id("a").bin_op(op, t.id("b")),
            )

    def test_binary_precedences(self):
        t = TestDocument("a + b * c")

        self.assertExprEqual(
            t.body,
            t.id("a") + t.id("b") * t.id("c"),
        )

    def test_list_comp(self):
        t = TestDocument("[x for x in [1, 2] if x > 1]")

        self.assertExprEqual(
            t.body,
            ListComp(
                range=t.body.range,
                expr=t.id("x"),
                for_spec=ForSpec(
                    range=t.range_of("for x in [1, 2]"),
                    id=t.id("x", nth=2),
                    expr=Array(
                        range=t.range_of("[1, 2]"),
                        values=[
                            t.num(1),
                            t.num(2),
                        ],
                    ),
                ),
                comp_spec=[
                    IfSpec(
                        range=t.range_of("if x > 1"),
                        condition=t.id("x", nth=3) > t.num(1, nth=2),
                    ),
                ],
            ),
        )

    def test_fn(self):
        t = TestDocument("function(x, y = 2) x + y")

        self.assertExprEqual(
            t.body,
            Fn(
                t.body.range,
                params=[
                    t.param("x"),
                    t.param("y", value=t.num(2)),
                ],
                body=t.id("x", nth=2) + t.id("y", nth=2),
            ),
        )

    def test_fn_no_params(self):
        t = TestDocument("function() 1")

        self.assertExprEqual(
            t.body,
            Fn(t.body.range, [], t.num(1)),
        )

    def test_import(self):
        t = TestDocument("import 'test.jsonnet'")

        self.assertExprEqual(
            t.body,
            Import(
                range=t.body.range,
                type="import",
                path=t.str(
                    value="test.jsonnet",
                    literal="'test.jsonnet'",
                ),
            ),
        )

    def test_importstr(self):
        t = TestDocument("importstr 'test.jsonnet'")

        self.assertExprEqual(
            t.body,
            Import(
                range=t.body.range,
                type="importstr",
                path=t.str(
                    value="test.jsonnet",
                    literal="'test.jsonnet'",
                ),
            ),
        )

    def test_assert_expr_without_message(self):
        t = TestDocument("assert true; false")

        self.assertExprEqual(
            t.body,
            AssertExpr(
                range=t.body.range,
                assertion=Assert(
                    range=t.range_of("assert true"),
                    condition=t.boolean(True),
                    message=None,
                ),
                body=t.boolean(False),
            ),
        )

    def test_assert_expr_with_message(self):
        t = TestDocument("assert true: 'never'; false")

        self.assertExprEqual(
            t.body,
            AssertExpr(
                range=t.body.range,
                assertion=Assert(
                    range=t.range_of("assert true: 'never'"),
                    condition=t.boolean(True),
                    message=t.str("never", "'never'"),
                ),
                body=t.boolean(False),
            ),
        )

    def test_assert_expr_in_bind(self):
        t = TestDocument("local x = assert true; false; x")

        self.assertExprEqual(
            t.body,
            Local(
                range=t.body.range,
                binds=[
                    t.id("x").bind(
                        AssertExpr(
                            range=t.range_of("assert true; false"),
                            assertion=Assert(
                                range=t.range_of("assert true"),
                                condition=t.boolean(True),
                            ),
                            body=t.boolean(False),
                        )
                    )
                ],
                body=t.id("x", nth=2),
            ),
        )

    def test_nested_assert_expr(self):
        # Assertions are right associated.
        t = TestDocument("assert true; assert false; x")

        self.assertExprEqual(
            t.body,
            AssertExpr(
                range=t.body.range,
                assertion=Assert(
                    range=t.range_of("assert true"),
                    condition=t.boolean(True),
                ),
                body=AssertExpr(
                    range=t.range_of("assert false; x"),
                    assertion=Assert(
                        range=t.range_of("assert false"),
                        condition=t.boolean(False),
                    ),
                    body=t.id("x"),
                ),
            ),
        )


@app.command()
def serve():
    server.start_io()


@app.command()
def test():
    unittest.main(argv=["--locals"], verbosity=1)


if __name__ == "__main__":
    app()

# vim:ft=python tw=88
