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

import dataclasses as D
import logging
import sys
import unittest
from collections import defaultdict
from contextlib import contextmanager
from enum import StrEnum
from itertools import accumulate, chain, groupby
from pathlib import Path
from textwrap import dedent
from typing import Annotated, Any, Iterable, Type, TypeVar, cast

import lsprotocol.types as L
import tree_sitter as T
import typer
from pygls.lsp.server import LanguageServer
from rich.console import Console
from rich.text import Text
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

log = logging.root

LANG_JSONNET = get_language("jsonnet")
JSONNET_TS_PARSER = get_parser("jsonnet")


ASTType = TypeVar("ASTType")


def strip_comments(nodes: list[T.Node]) -> list[T.Node]:
    return [node for node in nodes if not node.type == "comment"]


U = TypeVar("U")

Maybe = tuple[U] | tuple[()]


def maybe[U](v: U | None) -> Maybe[U]:
    return () if v is None else (v,)


def head_or_none[U](i: Iterable[U]) -> U | None:
    return next(iter(i), None)


@D.dataclass
class AST:
    location: L.Location

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "AST":
        global dispatch

        def skip_paren(uri: str, node: T.Node):
            return AST.from_doc(uri, strip_comments(node.named_children)[0])

        def dispatch_object(uri: str, node: T.Node):
            match strip_comments(node.named_children):
                case head, *_ if head.type == "objforloop":
                    return ObjComp.from_doc(uri, head)
                case _:
                    return Object.from_doc(uri, node)

        dispatch = {
            "anonymous_function": Fn.from_doc,
            "array": Array.from_doc,
            "assert": AssertExpr.from_doc,
            "binary": Binary.from_doc,
            "bind": Bind.from_doc,
            "conditional": If.from_doc,
            "document": Document.from_doc,
            "false": Bool.from_doc,
            "fieldaccess": FieldAccess.from_doc,
            "fieldname": FieldKey.from_doc,
            "field": Field.from_doc,
            "forloop": ListComp.from_doc,
            "forspec": ForSpec.from_doc,
            "functioncall": Call.from_doc,
            "id": Id.from_doc,
            "ifspec": IfSpec.from_doc,
            "implicit_plus": Binary.from_doc,
            "import": Import.from_doc,
            "importstr": Import.from_doc,
            "indexing": Slice.from_doc,
            "local_bind": Local.from_doc,
            "named_argument": Arg.from_doc,
            "number": Num.from_doc,
            "object": dispatch_object,
            "param": Param.from_doc,
            "parenthesis": skip_paren,
            "string": Str.from_doc,
            "true": Bool.from_doc,
        }

        constructor = dispatch.get(node.type, Unknown.from_doc)

        return constructor(uri, node)

    def to(self, expect_type: Type[ASTType]) -> ASTType:
        if not isinstance(self, expect_type):
            raise TypeError(
                f"Expected {expect_type.__name__}, but got {type(self).__name__}"
            )

        return cast(ASTType, self)

    @property
    def pretty_tree(self) -> str:
        return str(PrettyAST(self))


ESCAPE_TABLE: dict[int, str] = str.maketrans(
    {"\n": r"\n", "\t": r"\t", "\r": r"\r", '"': r"\""}
)


def escape(s: str, size: int = 50) -> str:
    escaped = s[0:size].translate(ESCAPE_TABLE)
    postfix = "" if len(s) <= size else f"[{len(s) - size} characters]"
    return f'"{escaped}{postfix}"'


class PrettyTree:
    """An abstract class for pretty-printing tree-like structures."""

    def node_text(self) -> str:
        """Returns a single-line string representing a tree node."""
        ...

    def children(self) -> list["PrettyTree"]:
        """Returns a list of child nodes."""
        ...

    def __repr__(self):
        def grow(buf: list[str], nodes: list[PrettyTree], branches: str = ""):
            for i, node in enumerate(nodes):
                last_child = i == len(nodes) - 1
                new_branch = ".   " if last_child else "|   "
                fork = "`-- " if last_child else "|-- "

                buf.append(f"{branches}{fork}{node.node_text()}")
                grow(buf, node.children(), branches + new_branch)

        lines = [self.node_text()]
        grow(lines, self.children())

        return "\n".join(lines)


@D.dataclass
class PrettyAST(PrettyTree):
    """A class for pretty-printing a Jsonnet AST."""

    node: Any
    label: str | None = None

    def node_text(self) -> str:
        match self.node:
            case Document() as doc:
                # For the top-level `Document` node, prints the full location with URI.
                repr = f"{doc.__class__.__name__} [{doc.location}]"
            case AST() as ast:
                # For all other nodes, only prints the range.
                repr = f"{ast.__class__.__name__} [{ast.location.range}]"
            case _, *_:
                # For lists, print a placeholder as all the elements are printed
                # separately as child nodes.
                repr = "[...]"
            case str():
                # Escapes (and truncates) strings, which can potentially be multi-line.
                repr = escape(self.node)
            case _:
                # Falls back to `__str__` for everything else.
                repr = str(self.node)

        # Prepends the label, if any.
        return repr if self.label is None else f"{self.label}={repr}"

    def children(self) -> list["PrettyTree"]:
        match self.node:
            case AST() as ast:
                return [
                    PrettyAST(getattr(ast, f.name), f.name)
                    for f in D.fields(ast)
                    if f.name != "location"
                ]
            case list() as array if (size := len(array)) > 0:
                return [PrettyAST(array[i], f"[{i}]") for i in range(size)]
            case _:
                return []

    def __repr__(self):
        return super().__repr__()


@D.dataclass
class PrettyCST(PrettyTree):
    """A class for pretty-printing a tree-sitter CST."""

    node: T.Node
    label: str | None = None

    def node_text(self) -> str:
        if not self.node.is_named and self.node.text:
            repr = f"{escape(self.node.text.decode())} [{range_of(self.node)}]"
        else:
            repr = f"{self.node.type} [{range_of(self.node)}]"

        return repr if self.label is None else f"{self.label}={repr}"

    def children(self) -> list["PrettyTree"]:
        return [
            PrettyCST(child, self.node.field_name_for_child(i))
            for i, child in enumerate(self.node.children)
        ]

    def __repr__(self):
        return super().__repr__()


@D.dataclass
class PrettyScope(PrettyTree):
    """A class for pretty-printing a scope."""

    node: Any
    label: str | None = None

    def node_text(self) -> str:
        match self.node:
            case Scope():
                repr = "Scope"
            case Binding(_, name, _, None):
                repr = name
            case Binding(_, name, _, value):
                repr = f'"{name}" <- {value.__class__.__name__}'
            case []:
                repr = "[]"
            case list():
                repr = "[...]"
            case _:
                repr = str(self.node)

        return repr if self.label is None else f"{self.label}={repr}"

    def children(self) -> list["PrettyTree"]:
        match self.node:
            case Scope(bindings, _, children):
                return [
                    PrettyScope(bindings, "bindings"),
                    PrettyScope(children, "children"),
                ]
            case list() as array:
                return [PrettyScope(value, f"[{i}]") for i, value in enumerate(array)]
            case _:
                return []

    def __repr__(self):
        return super().__repr__()


@D.dataclass
class Expr(AST):
    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Expr":
        return AST.from_doc(uri, node).to(Expr)

    def bin_op(self, op: "Operator", rhs: "Expr") -> "Binary":
        return Binary(merge_locations(self, rhs), op, self, rhs)

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


@D.dataclass
class Document(Expr):
    body: Expr

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Document":
        assert node.type == "document"
        body, *_ = strip_comments(node.named_children)
        return Document(location_of(uri, node), Expr.from_doc(uri, body))


@D.dataclass
class Id(Expr):
    name: str
    is_variable: bool = True

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Id":
        assert node.type == "id"
        assert node.text is not None
        return Id(location_of(uri, node), node.text.decode())

    def bind(self, value: Expr) -> "Bind":
        return Bind(merge_locations(self, value), self, value)

    def arg(self, value: Expr) -> "Arg":
        return Arg(merge_locations(self, value), value, self)

    @property
    def non_variable(self) -> "Id":
        return Id(self.location, self.name, False)


@D.dataclass
class Num(Expr):
    value: float

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Num":
        assert node.type == "number"
        assert node.text is not None
        return Num(location_of(uri, node), float(node.text.decode()))


@D.dataclass
class Str(Expr):
    raw: str

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Str":
        assert node.type == "string"

        _, content, _ = node.named_children
        assert content.text is not None

        return Str(
            location=location_of(uri, node),
            # The raw string content before escaping or indentations are handled.
            raw=content.text.decode(),
        )


@D.dataclass
class Bool(Expr):
    value: bool

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Bool":
        assert node.type in ["true", "false"]
        assert node.text is not None
        return Bool(location_of(uri, node), node.text.decode() == "true")


@D.dataclass
class Array(Expr):
    values: list[Expr]

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Array":
        assert node.type == "array"
        return Array(
            location=location_of(uri, node),
            values=[
                Expr.from_doc(uri, child)
                for child in strip_comments(node.named_children)
            ],
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


@D.dataclass
class Binary(Expr):
    op: Operator
    lhs: Expr
    rhs: Expr

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Binary":
        assert node.type in ["binary", "implicit_plus"]

        match node.type:
            case "binary":
                lhs, op, rhs, *_ = strip_comments(node.named_children)
                assert op.text is not None
                operator = Operator(op.text.decode())
            case _:
                lhs, rhs, *_ = strip_comments(node.named_children)
                operator = Operator.Plus

        return Binary(
            location=location_of(uri, node),
            op=operator,
            lhs=Expr.from_doc(uri, lhs),
            rhs=Expr.from_doc(uri, rhs),
        )


@D.dataclass
class Bind(AST):
    id: Id
    value: Expr

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Bind":
        assert node.type == "bind"

        children = strip_comments(node.named_children)
        assert len(children) >= 2

        if len(node.children_by_field_name("function")) == 0:
            id, value, *_ = children
            return Bind(
                location=location_of(uri, node),
                id=Id.from_doc(uri, id),
                value=Expr.from_doc(uri, value),
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
            id, *_ = children

            maybe_params = strip_comments(node.children_by_field_name("params"))
            assert len(maybe_params) <= 1

            body, *_ = strip_comments(node.children_by_field_name("body"))

            fn = Fn(
                location=location_of(uri, node),
                params=[
                    Param.from_doc(uri, param)
                    for params in maybe_params
                    for param in strip_comments(params.named_children)
                ],
                body=Expr.from_doc(uri, body),
            )

            return Bind(
                location=fn.location,
                id=Id.from_doc(uri, id),
                value=fn,
            )


@D.dataclass
class Local(Expr):
    binds: list[Bind]
    body: Expr

    def __post_init__(self):
        self.scope: Scope | None = None

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Local":
        assert node.type == "local_bind"

        binds, body = [], []

        # Drops the first node, which is the "local" keyword.
        for child in strip_comments(node.named_children)[1:]:
            (binds if child.type == "bind" else body).append(child)

        return Local(
            location=location_of(uri, node),
            binds=[Bind.from_doc(uri, bind) for bind in binds],
            body=Expr.from_doc(uri, body[0]),
        )


@D.dataclass
class Param(AST):
    id: Id
    default: Expr | None = None

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Param":
        assert node.type == "param"

        children = strip_comments(node.named_children)
        assert 1 <= len(children) <= 2

        id, *maybe_value = children
        return Param(
            location=location_of(uri, node),
            id=Id.from_doc(uri, id),
            default=head_or_none(Expr.from_doc(uri, value) for value in maybe_value),
        )


@D.dataclass
class Fn(Expr):
    params: list[Param]
    body: Expr

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Fn":
        assert node.type == "anonymous_function"
        first, *rest = strip_comments(node.named_children)

        match first, rest:
            case _, (body, *_) if first.type == "params":
                params = [
                    Param.from_doc(uri, param)
                    for param in strip_comments(first.named_children)
                ]
            case body, *_:
                params = []

        return Fn(
            location=location_of(uri, node),
            params=params,
            body=Expr.from_doc(uri, body),
        )


@D.dataclass
class Arg(AST):
    value: Expr
    id: Id | None = None

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Arg":
        if node.type == "named_argument":
            name, value = strip_comments(node.named_children)

            return Arg(
                location=location_of(uri, node),
                value=Expr.from_doc(uri, value),
                id=Id.from_doc(uri, name),
            )
        else:
            return Arg(
                location=location_of(uri, node),
                value=Expr.from_doc(uri, node),
            )


@D.dataclass
class Call(Expr):
    fn: Expr
    args: list[Arg]

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Call":
        assert node.type == "functioncall"

        children = strip_comments(node.named_children)
        assert len(children) >= 1

        fn, *maybe_args = children

        return Call(
            location=location_of(uri, node),
            fn=Expr.from_doc(uri, fn),
            args=[
                Arg.from_doc(uri, arg)
                for args in maybe_args
                for arg in strip_comments(args.named_children)
            ],
        )


@D.dataclass
class ForSpec(AST):
    id: Id
    expr: Expr

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "ForSpec":
        assert node.type == "forspec"
        id, expr = strip_comments(node.named_children)

        return ForSpec(
            location_of(uri, node),
            Id.from_doc(uri, id),
            Expr.from_doc(uri, expr),
        )


@D.dataclass
class IfSpec(AST):
    condition: Expr

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "IfSpec":
        assert node.type == "ifspec"
        [child] = strip_comments(node.named_children)
        return IfSpec(location_of(uri, node), condition=Expr.from_doc(uri, child))


@D.dataclass
class ListComp(Expr):
    expr: Expr
    for_spec: ForSpec
    comp_spec: list[ForSpec | IfSpec]

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "ListComp":
        assert node.type == "forloop"

        children = strip_comments(node.named_children)
        assert len(children) >= 2

        expr, for_spec, *maybe_comp_spec = children
        return ListComp(
            location=location_of(uri, node),
            expr=Expr.from_doc(uri, expr),
            for_spec=ForSpec.from_doc(uri, for_spec),
            comp_spec=[
                ForSpec.from_doc(uri, spec)
                if spec.type == "forspec"
                else IfSpec.from_doc(uri, spec)
                for comp_spec in maybe_comp_spec
                for spec in strip_comments(comp_spec.named_children)
            ],
        )


@D.dataclass
class Import(Expr):
    type: str
    path: Str

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Import":
        assert node.type in ["import", "importstr"]

        [path] = strip_comments(node.named_children)
        return Import(
            location_of(uri, node),
            node.type,
            Str.from_doc(uri, path),
        )


@D.dataclass
class Assert(AST):
    condition: Expr
    message: Expr | None = None

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Assert":
        assert node.type == "assert"

        children = strip_comments(node.named_children)
        assert 1 <= len(children) <= 2

        condition, *maybe_message = children

        match maybe_message:
            case child, *_:
                message = Expr.from_doc(uri, child)
            case _:
                message = None

        return Assert(
            location_of(uri, node),
            condition=Expr.from_doc(uri, condition),
            message=message,
        )


@D.dataclass
class AssertExpr(Expr):
    assertion: Assert
    body: Expr

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Expr":
        # Unfortunately, `tree-sitter-jsonnet` did a bad job in parsing assert
        # expressions (an expression following an assertion). Ideally, an `AssertExpr`
        # should map to a single tree-sitter node containing an assertion followed by an
        # expression. E.g., the given the following expression:
        #
        #   assert true: "never"; /* ! */ x
        #
        # it should be parsed into a single `<assert_expr>` node:
        #
        #   <assert_expr>     | assert true: "never"; /* ! */ x
        #       <assert>      | assert true: "never"
        #           <_expr>   |        true
        #           <string>  |              "never"
        #       <comment>     |                       /* ! */
        #       <id>          |                               x
        #
        # However, `tree-sitter-jsonnet` explicitly hides the top node by prefixing the
        # rule with an underscore (`_assert_expr` instead of `assert_expr`), and the
        # parser only returns the child nodes:
        #
        #   <assert>          | assert true: "never"
        #       <_expr>       |        true
        #       <string>      |              "never"
        #   <comment>         |                       /* ! */
        #   <id>              |                               x
        #
        # This unnecessarily complicates parsing values in local bindings and function
        # bodies, because a bound value and a function body can be either one node for a
        # regular expression or two nodes for an assertion expression.

        assertion = Assert.from_doc(uri, node)

        # Finds the first non-comment sibling named node. This is the first node of the
        # body expression.
        not_comment = node.next_named_sibling
        while not_comment is not None and not_comment.is_extra:
            not_comment = not_comment.next_named_sibling

        assert not_comment is not None
        body = Expr.from_doc(uri, not_comment)

        return AssertExpr(merge_locations(assertion, body), assertion, body)


@D.dataclass
class FieldKey(AST):
    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "FieldKey":
        assert node.type == "fieldname"

        location = location_of(uri, node)
        head, *tail = strip_comments(node.children)
        assert head.text is not None

        if head.text.decode() == "[":
            e, *_ = tail
            return DynamicKey(location, Expr.from_doc(uri, e))
        elif head.type == "id":
            return FixedKey(location, Id.from_doc(uri, head).non_variable)
        else:
            return FixedKey(location, Str.from_doc(uri, head))


@D.dataclass
class FixedKey(FieldKey):
    id: Id | Str


@D.dataclass
class DynamicKey(FieldKey):
    expr: Expr


class Visibility(StrEnum):
    Default = ":"
    Hidden = "::"
    Forced = ":::"


@D.dataclass
class Field(AST):
    key: FieldKey
    value: Expr
    visibility: Visibility = Visibility.Default
    inherited: bool = False

    def __post_init__(self):
        self.enclosing_self_scope: Scope | None = None

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Field":
        assert node.type == "field"
        children = strip_comments(node.children)

        if len(node.children_by_field_name("function")) == 0:
            key, plus_or_vis, *rest = children

            match plus_or_vis:
                case plus if plus.text == b"+":
                    vis, value, *_ = rest
                    assert vis.text is not None
                    return Field(
                        location=location_of(uri, node),
                        key=FieldKey.from_doc(uri, key),
                        value=Expr.from_doc(uri, value),
                        visibility=Visibility(vis.text.decode()),
                        inherited=True,
                    )
                case vis:
                    assert vis.text is not None
                    value, *_ = rest
                    return Field(
                        location=location_of(uri, node),
                        key=FieldKey.from_doc(uri, key),
                        value=Expr.from_doc(uri, value),
                        visibility=Visibility(vis.text.decode()),
                        inherited=False,
                    )
        else:
            key, _, params, _, vis, body, *_ = children
            assert vis.text is not None
            return Field(
                location=location_of(uri, node),
                key=FieldKey.from_doc(uri, key),
                value=Fn(
                    location_of(uri, node),
                    params=[
                        Param.from_doc(uri, param)
                        for param in strip_comments(params.named_children)
                    ],
                    body=Expr.from_doc(uri, body),
                ),
                visibility=Visibility(vis.text.decode()),
            )


@D.dataclass
class Object(Expr):
    binds: list[Bind] = D.field(default_factory=list)
    assertions: list[Assert] = D.field(default_factory=list)
    fields: list[Field] = D.field(default_factory=list)

    def __post_init__(self):
        self.self_scope: Scope | None = None
        self.super_scope: Scope | None = None

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Object":
        assert node.type == "object"

        binds = []
        assertions = []
        fields = []

        for member in strip_comments(node.named_children):
            assert member.type == "member"
            head, *_ = strip_comments(member.named_children)
            match head.type:
                case "objlocal":
                    _, bind, *_ = strip_comments(head.named_children)
                    binds.append(Bind.from_doc(uri, bind))
                case "assert":
                    assertions.append(Assert.from_doc(uri, head))
                case "field":
                    fields.append(Field.from_doc(uri, head))

        return Object(location_of(uri, node), binds, assertions, fields)


@D.dataclass
class ObjComp(Expr):
    binds: list[Bind]
    field: Field
    for_spec: ForSpec
    comp_spec: list[ForSpec | IfSpec] = D.field(default_factory=list)

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "ObjComp":
        assert node.type == "objforloop"
        field, for_spec, *maybe_comp_spec = node.named_children

        return ObjComp(
            location=location_of(uri, node),
            binds=[],
            field=Field.from_doc(uri, field),
            for_spec=ForSpec.from_doc(uri, for_spec),
            comp_spec=[
                ForSpec.from_doc(uri, spec)
                if spec.type == "forspec"
                else IfSpec.from_doc(uri, spec)
                for comp_spec in maybe_comp_spec
                for spec in strip_comments(comp_spec.named_children)
            ],
        )


@D.dataclass
class FieldAccess(Expr):
    obj: Expr
    field: Id

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "FieldAccess":
        assert node.type == "fieldaccess"
        expr, field = strip_comments(node.named_children)
        return FieldAccess(
            location=location_of(uri, node),
            obj=Expr.from_doc(uri, expr),
            field=Id.from_doc(uri, field).non_variable,
        )


@D.dataclass
class Slice(Expr):
    array: Expr
    begin: Expr
    end: Expr | None = None
    step: Expr | None = None

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Slice":
        assert node.type == "indexing"
        expr, begin, *rest = strip_comments(node.named_children)
        match rest:
            case [end, step]:
                end = Expr.from_doc(uri, end)
                step = Expr.from_doc(uri, step)
            case [end]:
                end = Expr.from_doc(uri, end)
                step = None
            case _:
                end = None
                step = None

        return Slice(
            location=location_of(uri, node),
            array=Expr.from_doc(uri, expr),
            begin=Expr.from_doc(uri, begin),
            end=end,
            step=step,
        )


@D.dataclass
class If(Expr):
    condition: Expr
    consequence: Expr
    alternative: Expr | None

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "If":
        assert node.type == "conditional"
        condition, consequence, *maybe_alternative = strip_comments(node.named_children)
        return If(
            location=location_of(uri, node),
            condition=Expr.from_doc(uri, condition),
            consequence=Expr.from_doc(uri, consequence),
            alternative=head_or_none(
                Expr.from_doc(uri, alternative) for alternative in maybe_alternative
            ),
        )


@D.dataclass
class Unknown(Expr):
    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Expr":
        return Unknown(location_of(uri, node))


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
            case If() as e:
                self.visit_if(e)
            case Object() as e:
                self.visit_object(e)
            case FieldAccess() as e:
                self.visit_field_access(e)

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
        self.visit_id(p.id)
        if p.default is not None:
            self.visit(p.default)

    def visit_call(self, e: Call):
        self.visit(e.fn)
        for a in e.args:
            self.visit_arg(a)

    def visit_arg(self, a: Arg):
        if a.id is not None:
            self.visit(a.id)
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

    def visit_if(self, e: If):
        self.visit(e.condition)
        self.visit(e.consequence)
        if e.alternative is not None:
            self.visit(e.alternative)

    def visit_object(self, e: Object):
        for f in e.fields:
            match f.key:
                case FixedKey() as key:
                    self.visit(key.id)
                case DynamicKey() as key:
                    self.visit(key.expr)

            self.visit_field(e, f)

        for b in e.binds:
            self.visit_bind(b)

        for a in e.assertions:
            self.visit_assert(a)

        for f in e.fields:
            self.visit(f.value)

    def visit_field(self, e: Object, f: Field):
        del e, f

    def visit_field_access(self, e: FieldAccess):
        self.visit(e.obj)
        self.visit(e.field)

    def visit_slice(self, e: Slice):
        self.visit(e.array)
        self.visit(e.begin)

        if e.end is not None:
            self.visit(e.end)

        if e.step is not None:
            self.visit(e.step)


def position_of(point: T.Point) -> L.Position:
    return L.Position(point.row, point.column)


def range_of(node: T.Node) -> L.Range:
    return L.Range(
        position_of(node.range.start_point),
        position_of(node.range.end_point),
    )


def location_of(uri: str, node: T.Node) -> L.Location:
    return L.Location(uri, range_of(node))


RangeLike = L.Range | T.Range | T.Node | AST


def merge_ranges(lhs: RangeLike, rhs: RangeLike) -> L.Range:
    match lhs:
        case L.Range() as r:
            start = r.start
        case T.Range() as r:
            start = position_of(r.start_point)
        case T.Node() as n:
            start = position_of(n.start_point)
        case AST() as a:
            start = a.location.range.start

    match rhs:
        case L.Range() as r:
            end = r.end
        case T.Range() as r:
            end = position_of(r.end_point)
        case T.Node() as n:
            end = position_of(n.end_point)
        case AST() as a:
            end = a.location.range.end

    assert start <= end

    return L.Range(start, end)


LocationLike = L.Location | AST


def merge_locations(lhs: LocationLike, rhs: LocationLike) -> L.Location:
    if isinstance(lhs, AST):
        lhs = lhs.location

    if isinstance(rhs, AST):
        rhs = rhs.location

    assert lhs.uri == rhs.uri, dedent(
        f"""\
        Cannot merge two locations from different documents:
        * {lhs.uri}
        * {rhs.uri}
        """
    )

    return L.Location(lhs.uri, merge_ranges(lhs.range, rhs.range))


def parse_jsonnet(source: str) -> T.Node:
    return JSONNET_TS_PARSER.parse(source.encode()).root_node


@D.dataclass
class Binding:
    scope: "Scope"
    name: str
    location: L.Location
    value: AST | None = None


@D.dataclass
class Scope:
    bindings: list[Binding] = D.field(default_factory=list)
    parent: "Scope | None" = None
    children: list["Scope"] = D.field(default_factory=list)

    def bind(self, name: str, location: L.Location, value: AST | None = None):
        self.bindings.insert(0, Binding(self, name, location, value))

    def get(self, id: Id) -> Binding | None:
        return next(
            iter(b for b in self.bindings if b.name == id.name),
            None if self.parent is None else self.parent.get(id),
        )

    def nest(self) -> "Scope":
        child = Scope([], parent=self)
        self.children.append(child)
        return child

    @property
    def pretty_tree(self) -> str:
        return str(PrettyScope(self))


@D.dataclass(frozen=True)
class HashablePosition:
    pos: L.Position

    def __hash__(self) -> int:
        return hash((self.pos.line, self.pos.character))


@D.dataclass(frozen=True)
class HashableLocation:
    location: L.Location

    def __hash__(self) -> int:
        return hash(
            (
                self.location.uri,
                self.location.range.start.line,
                self.location.range.start.character,
                self.location.range.end.line,
                self.location.range.end.character,
            )
        )

    def __lt__(self, other: "HashableLocation") -> bool:
        self_in_other = (
            other.location.range.start <= self.location.range.start
            and self.location.range.end <= other.location.range.end
        )

        return (
            self.location.uri < other.location.uri
            or self.location.uri == other.location.uri
            and self.location.range != other.location.range
            and (
                self_in_other
                or self.location.range.start <= other.location.range.start
                and self.location.range.end <= other.location.range.end
            )
        )


class DocumentIndex(Visitor):
    def __init__(self, workspace_index: "WorkspaceIndex", tree: Document) -> None:
        self.workspace_index = workspace_index
        self.tree: Document = tree
        self.inlay_hints: dict[HashablePosition, L.InlayHint] = {}
        self.current_scope: Scope = Scope()
        self.current_self: Scope | None = None
        self.current_super: Scope | None = None
        self.root_symbol = L.DocumentSymbol(
            name="__root__",
            kind=L.SymbolKind.File,
            range=tree.location.range,
            selection_range=tree.location.range,
        )
        self.breadcrumbs = [self.root_symbol]
        self.visit(tree)

    @property
    def uri(self):
        return self.tree.location.uri

    @property
    def document_symbols(self):
        return self.root_symbol.children

    def add_workspace_symbol(self, symbol: L.WorkspaceSymbol):
        self.workspace_index.workspace_symbols[self.uri].append(symbol)

    def add_document_symbol(self, symbol: L.DocumentSymbol):
        parent = self.breadcrumbs[-1]

        match parent.children:
            case list():
                parent.children.append(symbol)
            case None:
                parent.children = [symbol]

    def add_symbol(
        self,
        name: str,
        kind: L.SymbolKind,
        location: L.Location,
        selection_range: L.Range | None = None,
    ) -> tuple[L.WorkspaceSymbol, L.DocumentSymbol]:
        ws_symbol = L.WorkspaceSymbol(
            location=location,
            name=name,
            kind=kind,
        )

        doc_symbol = L.DocumentSymbol(
            name=name,
            kind=kind,
            range=location.range,
            selection_range=selection_range or location.range,
        )

        self.add_workspace_symbol(ws_symbol)
        self.add_document_symbol(doc_symbol)

        return ws_symbol, doc_symbol

    def add_id_reference(self, ref: Expr, binding: Binding):
        _, refs = self.workspace_index.add_reference(ref.location, binding.location)

        self.add_hint(
            L.InlayHint(
                position=ref.location.range.end,
                label="󰁝",
                kind=L.InlayHintKind.Parameter,
            )
        )

        # Shows the # of references.
        self.add_hint(
            L.InlayHint(
                position=binding.location.range.end,
                label=f"󰁅{len(refs)}",
                kind=L.InlayHintKind.Parameter,
            )
        )

    def add_hint(self, hint: L.InlayHint):
        self.inlay_hints[HashablePosition(hint.position)] = hint

    @contextmanager
    def parent_symbol(self, symbol: L.DocumentSymbol):
        self.breadcrumbs.append(symbol)
        try:
            yield symbol
        finally:
            self.breadcrumbs.pop()

    @contextmanager
    def nested_scope(self):
        prev = self.current_scope
        self.current_scope = self.current_scope.nest()
        try:
            yield self.current_scope
        finally:
            self.current_scope = prev

    @contextmanager
    def super_scope(self, scope: Scope | None):
        prev = self.current_super
        self.current_super = scope
        try:
            yield self.current_super
        finally:
            self.current_super = prev

    @contextmanager
    def self_scope(self, scope: Scope | None):
        prev = self.current_self
        self.current_self = scope
        try:
            yield self.current_self
        finally:
            self.current_self = prev

    def find_object_scope(
        self,
        t: AST,
        scope: Scope,
        self_scope: Scope | None = None,
        super_scope: Scope | None = None,
    ) -> Scope | None:
        # Not used yet
        del super_scope

        match t:
            case Object() as obj if obj.self_scope is not None:
                return obj.self_scope
            case Document():
                return self.find_object_scope(t.body, Scope())
            case Id() as id:
                for binding in maybe(scope.get(id)):
                    for value in maybe(binding.value):
                        return self.find_object_scope(value, binding.scope)
            case Field() as f:
                for self_scope in maybe(f.enclosing_self_scope):
                    return self.find_object_scope(f.value, scope, self_scope)
            case FieldAccess() as f:
                for parent_scope in maybe(self.find_object_scope(f.obj, scope)):
                    for binding in maybe(parent_scope.get(f.field)):
                        for value in maybe(binding.value):
                            return self.find_object_scope(value, binding.scope)
            case Binary(_, _, _, rhs):
                return self.find_object_scope(rhs, self.current_scope)
            case Import(_, "import", path):
                for index in maybe(self.importee_index(path.raw)):
                    return self.find_object_scope(index.tree, index.current_scope)
            case Local() as l:
                for local_scope in maybe(l.scope):
                    return self.find_object_scope(l.body, local_scope)
            case Binary(_, Operator.Plus, _, rhs):
                return self.find_object_scope(rhs, scope)
            case _:
                return None

        return None

    def visit_local(self, e: Local):
        for b in e.binds:
            self.visit_bind(b)

        with self.nested_scope() as nested:
            e.scope = nested
            self.visit(e.body)

    def visit_bind(self, b: Bind):
        self.current_scope.bind(b.id.name, b.id.location, b.value)

        match b.value:
            case Fn():
                kind = L.SymbolKind.Function
            case _:
                kind = L.SymbolKind.Variable

        _, doc_symbol = self.add_symbol(
            name=b.id.name,
            kind=kind,
            location=b.id.location,
            selection_range=b.location.range,
        )

        with self.parent_symbol(doc_symbol):
            with self.nested_scope():
                self.visit(b.value)

    def visit_fn(self, e: Fn):
        # NOTE: In a Jsonnet function, any parameter's default value expression can
        # reference any other peer parameters, e.g.:
        #
        #   local f(x = y, y, x = z) =
        #       x + y + z;
        #   f(y = 2)
        #
        # This requires all parameters to be bound before traversing any parameter
        # default value expressions. This is also why parameters must be handled in
        # `visit_fn` instead of `visit_param`.
        with self.nested_scope():
            for p in e.params:
                self.current_scope.bind(p.id.name, p.id.location, p.default)

            for p in e.params:
                _, doc_symbol = self.add_symbol(
                    p.id.name,
                    L.SymbolKind.Variable,
                    p.id.location,
                    p.location.range,
                )

                if p.default is not None:
                    with self.parent_symbol(doc_symbol):
                        self.visit(p.default)

            self.visit(e.body)

    def visit_for_spec(self, s: ForSpec):
        self.visit(s.expr)

        symbol = L.DocumentSymbol(
            name=s.id.name,
            kind=L.SymbolKind.Variable,
            range=s.id.location.range,
            selection_range=s.location.range,
        )

        with self.parent_symbol(symbol):
            pass

    def visit_id(self, e: Id):
        if e.is_variable and (binding := self.current_scope.get(e)) is not None:
            self.add_id_reference(e, binding)

    def visit_field(self, e: Object, f: Field):
        assert e.self_scope is not None

        f.enclosing_self_scope = e.self_scope

        match f.key:
            case FixedKey(_, Id(_, name)):
                e.self_scope.bind(name, f.location, f)
            case FixedKey(_, Str(_, raw)):
                e.self_scope.bind(raw, f.location, f)

        # Adds an optional inlay hint for visibility and inheritance.
        label_parts = []

        match f.visibility:
            case Visibility.Hidden:
                label_parts.append("hidden")
            case Visibility.Forced:
                label_parts.append("forced visible")

        if f.inherited:
            label_parts.append("inherited")

        if len(label_parts) > 0:
            self.add_hint(
                L.InlayHint(
                    position=f.key.location.range.end,
                    label=", ".join(label_parts),
                    kind=L.InlayHintKind.Parameter,
                    padding_left=True,
                    padding_right=True,
                )
            )

    def resolve_importee_path(self, raw_path: str) -> Path:
        root = Path.from_uri(self.workspace_index.root_uri)
        return root.joinpath(raw_path).absolute()

    def importee_index(self, raw_path: str) -> "DocumentIndex | None":
        path = self.resolve_importee_path(raw_path)
        index = self.workspace_index.docs.get(path.as_uri())

        return index or (
            self.workspace_index.get_or_sync(
                uri=path.as_uri(),
                source=path.read_text(encoding="utf-8"),
            )
            if path.exists()
            else None
        )

    def visit_import(self, e: Import):
        self.add_symbol(
            name=e.path.raw,
            kind=L.SymbolKind.File,
            location=e.path.location,
            selection_range=e.location.range,
        )

        self.add_hint(
            L.InlayHint(
                position=e.path.location.range.end,
                label="",
                kind=L.InlayHintKind.Parameter,
            )
        )

        for index in maybe(self.importee_index(e.path.raw)):
            self.workspace_index.add_reference(e.path.location, index.tree.location)

    def visit_object(self, e: Object):
        e.self_scope = Scope(parent=self.current_super)
        with self.self_scope(e.self_scope):
            super().visit_object(e)

    def visit_field_access(self, e: FieldAccess):
        self.visit(e.obj)

        for scope in maybe(self.find_object_scope(e.obj, self.current_scope)):
            for binding in maybe(scope.get(e.field)):
                self.add_id_reference(e.field, binding)

    def visit_binary(self, e: Binary):
        self.visit(e.lhs)

        lhs_scope = self.find_object_scope(e.lhs, self.current_scope)
        with self.super_scope(lhs_scope):
            self.visit(e.rhs)


LocationMap = dict[HashableLocation, list[L.Location]]


class WorkspaceIndex:
    def __init__(self, root_uri: str):
        self.root_uri = root_uri

        # Indexed documents
        self.docs: dict[str, DocumentIndex] = {}

        # TODO: Use a suffix tree to make it scalable.
        self.workspace_symbols: dict[str, list[L.WorkspaceSymbol]] = defaultdict(
            list[L.WorkspaceSymbol]
        )

        # For go-to-definition and find-references
        self.ref_to_defs: LocationMap = defaultdict(list[L.Location])
        self.def_to_refs: LocationMap = defaultdict(list[L.Location])

    def find_linked_locations(
        self, uri: str, pos: L.Position, map: LocationMap
    ) -> list[L.Location]:
        # TODO: More efficient position-to-location search.

        for key, locations in sorted(map.items(), key=lambda pair: pair[0]):
            r = key.location.range
            if uri == key.location.uri and r.start <= pos <= r.end:
                return locations

        return []

    def add_reference(
        self, ref_location: L.Location, def_location: L.Location
    ) -> tuple[list[L.Location], list[L.Location]]:
        defs = self.ref_to_defs[HashableLocation(ref_location)]
        refs = self.def_to_refs[HashableLocation(def_location)]

        defs.append(def_location)
        refs.append(ref_location)

        return defs, refs

    def definitions(self, uri: str, position: L.Position) -> list[L.Location]:
        return self.find_linked_locations(uri, position, self.ref_to_defs)

    def references(self, uri: str, position: L.Position) -> list[L.Location]:
        return self.find_linked_locations(uri, position, self.def_to_refs)

    def sync(self, uri: str, source: str):
        cst = parse_jsonnet(source)
        tree = Expr.from_doc(uri, cst)
        self.docs[uri] = DocumentIndex(self, tree.to(Document))

    def invalidate_location_map(self, uri: str, map: LocationMap) -> LocationMap:
        map = defaultdict(
            list[L.Location],
            {
                key: value
                for key, value in self.ref_to_defs.items()
                if key.location.uri != uri
            },
        )

        for key, locations in map.items():
            map[key] = [loc for loc in locations if loc.uri != uri]

        return map

    def invalidate_doc(self, uri: str):
        self.ref_to_defs = self.invalidate_location_map(uri, self.ref_to_defs)
        self.def_to_refs = self.invalidate_location_map(uri, self.def_to_refs)

        if uri in self.workspace_symbols:
            del self.workspace_symbols[uri]

        if uri in self.docs:
            del self.docs[uri]

    def get_or_sync(self, uri: str, source: str) -> DocumentIndex:
        if uri not in self.docs:
            self.sync(uri, source)
        return self.docs[uri]


class JustLanguageServer(LanguageServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_workspace_root(self, root_path: str, root_uri: str):
        self.workspace.add_folder(L.WorkspaceFolder(root_path, root_uri))
        self.workspace_index = WorkspaceIndex(root_uri)


server = JustLanguageServer("just", "v0.1")


@server.feature(L.INITIALIZE)
def initialize(ls: JustLanguageServer, params: L.InitializeParams):
    for root_path in maybe(params.root_path):
        for root_uri in maybe(params.root_uri):
            log.info("Discovered workspace root: %s, %s", root_path, root_uri)

            root = Path.from_uri(root_uri).absolute()
            assert root.as_posix() == Path(root_path).absolute().as_posix()

            ls.set_workspace_root(root.as_posix(), root.as_uri())

    return L.InitializeResult(
        capabilities=L.ServerCapabilities(
            definition_provider=True,
            document_symbol_provider=True,
            inlay_hint_provider=True,
            references_provider=True,
            text_document_sync=L.TextDocumentSyncKind.Full,
            workspace_symbol_provider=True,
        ),
        server_info=L.ServerInfo(
            name="just",
            version="v0.1",
        ),
    )


@server.feature(L.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: JustLanguageServer, params: L.DidOpenTextDocumentParams):
    doc = params.text_document

    # If the workspace root is not yet discovered, set the root as the parent folder of
    # the first document opened.
    if len(ls.workspace.folders) == 0:
        root = Path.from_uri(doc.uri).absolute().parent
        ls.set_workspace_root(root.as_posix(), root.as_uri())

    ls.workspace_index.sync(doc.uri, doc.text)


@server.feature(L.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: JustLanguageServer, params: L.DidChangeTextDocumentParams):
    # TODO: Patch the AST incrementally.
    doc = ls.workspace.get_text_document(params.text_document.uri)
    ls.workspace_index.sync(doc.uri, doc.source)


@server.feature(L.WORKSPACE_SYMBOL)
def workspace_symbol(ls: JustLanguageServer, params: L.WorkspaceSymbolParams):
    return [
        symbol
        for symbols in ls.workspace_index.workspace_symbols.values()
        for symbol in symbols
        if params.query in symbol.name
    ]


@server.feature(L.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
def document_symbol(ls: JustLanguageServer, params: L.DocumentColorParams):
    doc = ls.workspace.get_text_document(params.text_document.uri)
    return ls.workspace_index.get_or_sync(doc.uri, doc.source).document_symbols


@server.feature(L.TEXT_DOCUMENT_DEFINITION)
def definition(ls: JustLanguageServer, params: L.DefinitionParams):
    return ls.workspace_index.definitions(params.text_document.uri, params.position)


@server.feature(L.TEXT_DOCUMENT_REFERENCES)
def references(ls: JustLanguageServer, params: L.ReferenceParams):
    return ls.workspace_index.references(params.text_document.uri, params.position)


@server.feature(L.TEXT_DOCUMENT_INLAY_HINT)
def inlay_hint(ls: JustLanguageServer, params: L.InlayHintParams):
    doc = ls.workspace.get_text_document(params.text_document.uri)
    doc_index = ls.workspace_index.get_or_sync(doc.uri, doc.source)
    return list(doc_index.inlay_hints.values())


@app.command()
def serve():
    server.start_io()


@app.command()
def tree(
    path: Annotated[
        Path,
        typer.Argument(
            help="The Jsonnet file to print.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            writable=False,
            allow_dash=True,
        ),
    ],
    tree_sitter: Annotated[
        bool,
        typer.Option(
            "-t",
            "--tree-sitter",
            help="Print the tree-sitter tree.",
        ),
    ] = False,
):
    if path == Path("-"):
        uri = "/dev/stdin"
        source = sys.stdin.read()
    else:
        uri = path.absolute().as_uri()
        source = path.read_text()

    cst = parse_jsonnet(source)
    ast = AST.from_doc(uri, cst)
    tree = PrettyCST(cst) if tree_sitter else PrettyAST(ast)

    Console(markup=False).print(tree)


@app.command()
def index(
    workspace_root: Annotated[
        Path,
        typer.Argument(
            help="The workspace root directory.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            allow_dash=False,
        ),
    ],
    path: Annotated[
        Path,
        typer.Argument(
            help="The Jsonnet file to index",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            allow_dash=False,
        ),
    ],
):
    WorkspaceIndex(workspace_root.as_uri()).sync(path.as_uri(), path.read_text())


@app.command(context_settings=dict(allow_extra_args=True))
def test(context: typer.Context):
    # `unittest` requires the first argument to be the program name, while Typer strips
    # the program (`just`) and command (`test`) names.
    argv = ["test"] + context.args
    unittest.main(argv=argv)


class TestDocument:
    def __init__(self, source: str, uri: str = "file:///tmp/test.jsonnet") -> None:
        self.uri = uri
        self.source = source
        self.root = parse_jsonnet(source)
        self.body = Document.from_doc(self.uri, self.root).body

        self.lines = source.splitlines(keepends=True)

        # Appends the last empty line if needed.
        if source.endswith("\n"):
            self.lines.append("")

        # Computes the character offset of the first character in each line, used for
        # converting line-character positions to offsets.
        self.line_offsets = list(accumulate(chain([0], map(len, self.lines))))

    def query_one(self, query: T.Query, capture: str) -> AST:
        [node] = T.QueryCursor(query).captures(self.root).get(capture, [])
        return AST.from_doc(self.uri, node)

    def offset_of(self, pos: L.Position) -> int:
        return self.line_offsets[pos.line] + pos.character

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

    def location_of(self, needle: str, line=1, nth=1) -> L.Location:
        start = self.start_of(needle, line, nth)
        end = L.Position(start.line, start.character + len(needle))
        return L.Location(self.uri, L.Range(start, end))

    def id(self, name: str, line=1, nth=1) -> Id:
        return Id(self.location_of(name, line, nth), name)

    def boolean(self, value: bool, line=1, nth=1) -> Bool:
        needle = "true" if value else "false"
        range = self.location_of(needle, line, nth)
        return Bool(range, value)

    def num(self, value: float | int, literal: str | None = None, line=1, nth=1) -> Num:
        match value, literal:
            case int(), None:
                literal = str(value)
                value = float(value)
            case _:
                assert literal is not None

        return Num(self.location_of(literal, line, nth), value)

    def str(self, value: str, literal: str, line=1, nth=1) -> Str:
        return Str(self.location_of(literal, line, nth), value)

    def param(
        self, name: str, line: int = 1, nth: int = 1, default: Expr | None = None
    ) -> Param:
        id = self.id(name, line, nth)
        location = id.location if default is None else merge_locations(id, default)
        return Param(location, id, default)

    def arg(self, value: Expr) -> Arg:
        return Arg(value.location, value)

    def highlight(self, ranges: list[L.Range], style: str):
        styled = Text.styled
        rendered_lines = []

        uri_line = styled(self.uri, "grey50")
        rendered_lines.append(uri_line)

        # Renders the ranges.
        rendered_source = styled(self.source, "default")
        for r in ranges:
            rendered_source.stylize(
                style,
                start=self.offset_of(r.start),
                end=self.offset_of(r.end),
            )

        raw_lines = self.source.splitlines()
        width = max(map(len, raw_lines))
        height = len(raw_lines)
        gutter_width = len(str(height))

        # Renders a horizontal ruler like the following:
        #
        #     0    5   10   15      <-- header line
        #     |''''|''''|''''|''''  <-- guide line
        #   1 |local x = { f: 1 };
        #   2 |local y = x.f;
        #   3 |x
        def render_ruler(width: int, left_padding: int) -> list[Text]:
            # To build the header line, builds right-aligned 5-character wide column
            # segments first, joins them together, then chops off the first 4 spaces.
            # The final left-padding is for the line number gutter.
            every_5_chars = range(0, width // 5 * 5 + 1, 5)
            header_segs = [f"{i:>5}" for i in every_5_chars]
            header_line = styled("".join(header_segs)[4:], "grey50")
            header_line.pad_left(left_padding)

            guide_line = styled(("|''''" * (width // 5 + 1))[: width + 1], "grey50")
            guide_line.pad_left(left_padding)

            return [header_line, guide_line]

        # Renders a top horizontal ruler.
        ruler_lines = render_ruler(width, gutter_width + 1)
        rendered_lines.extend(ruler_lines)

        # Renders source lines with line numbers.
        for i, line in enumerate(rendered_source.split()):
            line_no = styled(f"{i + 1:>{gutter_width}} |", "grey50")
            rendered_lines.append(line_no + line)

        # Renders a bottom horizontal ruler for long files.
        if height > 5:
            rendered_lines.extend(ruler_lines)

        return Text("\n").join(rendered_lines)


class TestAST(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.maxDiff = None

    def assertAstEqual(self, tree_or_source: AST | str, expected: AST | str):
        match tree_or_source, expected:
            case AST() as tree, str():
                self.assertMultiLineEqual(tree.pretty_tree, expected.strip())
            case AST() as tree, AST() if tree != expected:
                self.assertAstEqual(tree, expected.pretty_tree)
            case str() as source, AST():
                self.assertAstEqual(TestDocument(source).body, expected.pretty_tree)
            case str() as source, str():
                self.assertAstEqual(TestDocument(source).body, expected)

    def test_pretty_tree(self):
        self.assertAstEqual(
            "1",
            dedent(
                """\
                Num [0:0-0:1]
                `-- value=1.0
                """
            ),
        )

        self.assertAstEqual(
            "'f' + 1",
            dedent(
                """\
                Binary [0:0-0:7]
                |-- op="+"
                |-- lhs=Str [0:0-0:3]
                |   `-- raw="f"
                `-- rhs=Num [0:6-0:7]
                .   `-- value=1.0
                """
            ),
        )

        self.assertAstEqual(
            "[1, 2, 3]",
            dedent(
                """\
                Array [0:0-0:9]
                `-- values=[...]
                .   |-- [0]=Num [0:1-0:2]
                .   |   `-- value=1.0
                .   |-- [1]=Num [0:4-0:5]
                .   |   `-- value=2.0
                .   `-- [2]=Num [0:7-0:8]
                .   .   `-- value=3.0
                """,
            ),
        )

        self.assertAstEqual(
            "local x = 1; x + 2",
            dedent(
                """\
                Local [0:0-0:18]
                |-- binds=[...]
                |   `-- [0]=Bind [0:6-0:11]
                |   .   |-- id=Id [0:6-0:7]
                |   .   |   |-- name="x"
                |   .   |   `-- is_variable=True
                |   .   `-- value=Num [0:10-0:11]
                |   .   .   `-- value=1.0
                `-- body=Binary [0:13-0:18]
                .   |-- op="+"
                .   |-- lhs=Id [0:13-0:14]
                .   |   |-- name="x"
                .   |   `-- is_variable=True
                .   `-- rhs=Num [0:17-0:18]
                .   .   `-- value=2.0
                """
            ),
        )

    def test_number(self):
        t = TestDocument("1")

        self.assertAstEqual(
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
            self.assertAstEqual(
                t.body,
                Str(t.body.location, expected),
            )

    def test_paren(self):
        t = TestDocument("(1)")
        self.assertAstEqual(t.body, t.num(1))

        t = TestDocument("(assert true; 1)")
        self.assertAstEqual(
            t.body,
            AssertExpr(
                location=t.location_of("assert true; 1"),
                assertion=Assert(
                    location=t.location_of("assert true"),
                    condition=t.boolean(True),
                ),
                body=t.num(1),
            ),
        )

    def test_local(self):
        t = TestDocument("local x = 1; x")

        self.assertAstEqual(
            t.body,
            Local(
                location=t.body.location,
                binds=[t.id("x").bind(t.num(1))],
                body=t.id("x", nth=2),
            ),
        )

    def test_local_with_asserts(self):
        t = TestDocument(
            dedent(
                """\
                local v1 = 0;
                assert true;
                assert true;
                v1 + 2
                """
            )
        )

        inner_assert_expr = AssertExpr(
            location=merge_locations(
                t.location_of("assert", line=3),
                t.location_of("v1 + 2", line=4),
            ),
            assertion=Assert(
                location=t.location_of("assert true", line=3),
                condition=t.boolean(True, line=3),
            ),
            body=t.id("v1", line=4) + t.num(2, line=4),
        )

        outer_assert_expr = AssertExpr(
            location=merge_locations(
                t.location_of("assert", line=2),
                t.location_of("v1 + 2", line=4),
            ),
            assertion=Assert(
                location=t.location_of("assert true", line=2),
                condition=t.boolean(True, line=2),
            ),
            body=inner_assert_expr,
        )

        self.assertAstEqual(
            t.body,
            Local(
                location=t.body.location,
                binds=[
                    t.id("v1").bind(t.num(0)),
                ],
                body=outer_assert_expr,
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

        self.assertAstEqual(
            t.body,
            Local(
                location=t.body.location,
                binds=[
                    t.id("f1", line=2).bind(
                        Fn(
                            location=t.location_of("f1(x) = x + 1", line=2),
                            params=[t.param("x", line=2)],
                            body=t.id("x", line=2, nth=2) + t.num(1, line=2, nth=2),
                        )
                    ),
                    t.id("f2", line=3).bind(
                        Fn(
                            location=t.location_of("f2(y, z) = y + z", line=3),
                            params=[
                                t.param("y", line=3),
                                t.param("z", line=3),
                            ],
                            body=t.id("y", line=3, nth=2) + t.id("z", line=3, nth=2),
                        )
                    ),
                ],
                body=Call(
                    location=t.location_of("f2(f1(3), z = 4)", line=4),
                    fn=t.id("f2", line=4),
                    args=[
                        t.arg(
                            Call(
                                location=t.location_of("f1(3)", line=4),
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

        self.assertAstEqual(
            t.body,
            Local(
                location=t.body.location,
                binds=[
                    t.id("x").bind(t.num(1)),
                    t.id("y").bind(t.num(2)),
                ],
                body=t.id("x", line=2) + t.id("y", line=2),
            ),
        )

    def test_empty_array(self):
        t = TestDocument("[]")

        self.assertAstEqual(
            t.body,
            Array(location=t.body.location, values=[]),
        )

    def test_array(self):
        t = TestDocument("[1, true, /* ! */ '3']")

        self.assertAstEqual(
            t.body,
            Array(
                location=t.body.location,
                values=[
                    t.num(1),
                    t.boolean(True),
                    t.str("3", literal="'3'"),
                ],
            ),
        )

    def test_binary(self):
        for op in Operator:
            t = TestDocument(f"a /*!*/ {op.value} /*!*/ b")

            self.assertAstEqual(
                t.body,
                t.id("a").bin_op(op, t.id("b")),
            )

    def test_implicit_plus(self):
        t = TestDocument("a {}")

        self.assertAstEqual(
            t.body,
            t.id("a") + Object(t.location_of("{}")),
        )

    def test_binary_precedences(self):
        t = TestDocument("a + b * c")

        self.assertAstEqual(
            t.body,
            t.id("a") + (t.id("b") * t.id("c")),
        )

        t = TestDocument("(a + b) * c")

        self.assertAstEqual(
            t.body,
            Binary(
                t.body.location,
                op=Operator.Multiply,
                lhs=t.id("a") + t.id("b"),
                rhs=t.id("c"),
            ),
        )

    def test_list_comp(self):
        t = TestDocument("[x for x in [1, 2] if x > 1]")

        self.assertAstEqual(
            t.body,
            ListComp(
                location=t.body.location,
                expr=t.id("x"),
                for_spec=ForSpec(
                    location=t.location_of("for x in [1, 2]"),
                    id=t.id("x", nth=2),
                    expr=Array(
                        location=t.location_of("[1, 2]"),
                        values=[
                            t.num(1),
                            t.num(2),
                        ],
                    ),
                ),
                comp_spec=[
                    IfSpec(
                        location=t.location_of("if x > 1"),
                        condition=t.id("x", nth=3) > t.num(1, nth=2),
                    ),
                ],
            ),
        )

    def test_fn(self):
        t = TestDocument("function(x, y = 2) x + y")

        self.assertAstEqual(
            t.body,
            Fn(
                t.body.location,
                params=[
                    t.param("x"),
                    t.param("y", default=t.num(2)),
                ],
                body=t.id("x", nth=2) + t.id("y", nth=2),
            ),
        )

    def test_fn_no_params(self):
        t = TestDocument("function() 1")

        self.assertAstEqual(
            t.body,
            Fn(t.body.location, [], t.num(1)),
        )

    def test_import(self):
        t = TestDocument("import 'test.jsonnet'")

        self.assertAstEqual(
            t.body,
            Import(
                location=t.body.location,
                type="import",
                path=t.str(
                    value="test.jsonnet",
                    literal="'test.jsonnet'",
                ),
            ),
        )

    def test_importstr(self):
        t = TestDocument("importstr 'test.jsonnet'")

        self.assertAstEqual(
            t.body,
            Import(
                location=t.body.location,
                type="importstr",
                path=t.str(
                    value="test.jsonnet",
                    literal="'test.jsonnet'",
                ),
            ),
        )

    def test_assert_expr_without_message(self):
        t = TestDocument("assert true; false")

        self.assertAstEqual(
            t.body,
            AssertExpr(
                location=t.body.location,
                assertion=Assert(
                    location=t.location_of("assert true"),
                    condition=t.boolean(True),
                    message=None,
                ),
                body=t.boolean(False),
            ),
        )

    def test_assert_expr_with_message(self):
        t = TestDocument("assert true: 'never'; /*!*/ false")

        self.assertAstEqual(
            t.body,
            AssertExpr(
                location=t.body.location,
                assertion=Assert(
                    location=t.location_of("assert true: 'never'"),
                    condition=t.boolean(True),
                    message=t.str("never", literal="'never'"),
                ),
                body=t.boolean(False),
            ),
        )

    def test_assert_expr_in_bind(self):
        t = TestDocument("local x = assert true; false; x")

        self.assertAstEqual(
            t.body,
            Local(
                location=t.body.location,
                binds=[
                    t.id("x").bind(
                        AssertExpr(
                            location=t.location_of("assert true; false"),
                            assertion=Assert(
                                location=t.location_of("assert true"),
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

        self.assertAstEqual(
            t.body,
            AssertExpr(
                location=t.body.location,
                assertion=Assert(
                    location=t.location_of("assert true"),
                    condition=t.boolean(True),
                ),
                body=AssertExpr(
                    location=t.location_of("assert false; x"),
                    assertion=Assert(
                        location=t.location_of("assert false"),
                        condition=t.boolean(False),
                    ),
                    body=t.id("x"),
                ),
            ),
        )

    def assertAstEqualByQuery(
        self, doc: TestDocument, query: T.Query, capture: str, expected: AST
    ):
        captures = T.QueryCursor(query).captures(doc.root)
        [node] = captures[capture]
        self.assertAstEqual(AST.from_doc(doc.uri, node), expected)

    object_query = T.Query(
        LANG_JSONNET,
        dedent(
            """\
            (object
              (member
                (field
                  (fieldname) @field_key
                  (_)*) @field)) @object

            (object
              (objforloop)) @object
            """
        ),
    )

    def test_object_field_name(self):
        t = TestDocument("local x = 'f'; { [x]: 1 }")

        self.assertAstEqual(
            t.query_one(self.object_query, "field_key"),
            DynamicKey(
                t.location_of("[x]"),
                t.id("x", nth=2),
            ),
        )

        t = TestDocument("{ x: 1 }")

        self.assertAstEqual(
            t.query_one(self.object_query, "field_key"),
            FixedKey(
                t.location_of("x"),
                t.id("x").non_variable,
            ),
        )

        t = TestDocument("{ 'x': 1 }")

        self.assertAstEqual(
            t.query_one(self.object_query, "field_key"),
            FixedKey(
                t.location_of("'x'"),
                t.str("x", literal="'x'"),
            ),
        )

    def test_object_field(self):
        t = TestDocument("{ x: 1 }")

        self.assertAstEqual(
            t.query_one(self.object_query, "field"),
            Field(
                location=t.location_of("x: 1"),
                key=FixedKey(t.location_of("x"), t.id("x").non_variable),
                value=t.num(1),
            ),
        )

        t = TestDocument("{ x+::: 1 }")

        self.assertAstEqual(
            t.query_one(self.object_query, "field"),
            Field(
                location=t.location_of("x+::: 1"),
                key=FixedKey(t.location_of("x"), t.id("x").non_variable),
                value=t.num(1),
                visibility=Visibility.Forced,
                inherited=True,
            ),
        )

        t = TestDocument("{ f(p1, p2 = 0):: p1 + p2 }")

        self.assertAstEqual(
            t.query_one(self.object_query, "field"),
            Field(
                location=t.location_of("f(p1, p2 = 0):: p1 + p2"),
                key=FixedKey(t.location_of("f"), t.id("f").non_variable),
                value=Fn(
                    t.location_of("f(p1, p2 = 0):: p1 + p2"),
                    params=[
                        t.param("p1"),
                        t.param("p2", default=t.num(0)),
                    ],
                    body=t.id("p1", nth=2) + t.id("p2", nth=2),
                ),
                visibility=Visibility.Hidden,
            ),
        )

    def test_obj_comp(self):
        t = TestDocument("{ ['f' + x]: 0 for x in [1, 2] if x > 1 }")

        self.assertAstEqual(
            t.query_one(self.object_query, "object"),
            ObjComp(
                location=t.body.location,
                binds=[],
                field=Field(
                    location=t.location_of("['f' + x]: 0"),
                    key=DynamicKey(
                        t.location_of("['f' + x]"),
                        t.str("f", literal="'f'") + t.id("x"),
                    ),
                    value=t.num(0),
                ),
                for_spec=ForSpec(
                    location=t.location_of("for x in [1, 2]"),
                    id=t.id("x", nth=2),
                    expr=Array(
                        location=t.location_of("[1, 2]"),
                        values=[t.num(1), t.num(2)],
                    ),
                ),
                comp_spec=[
                    IfSpec(
                        location=t.location_of("if x > 1"),
                        condition=t.id("x", nth=3) > t.num(1, nth=2),
                    )
                ],
            ),
        )


class TestWorkspace:
    def __init__(self, root_uri: str, docs: list[TestDocument]) -> None:
        self.docs = {doc.uri: doc for doc in docs}
        self.index = WorkspaceIndex(root_uri)

        for doc in docs:
            assert doc.uri not in self.index.docs
            self.index.sync(doc.uri, doc.source)

    @staticmethod
    def single_doc(doc: TestDocument) -> "TestWorkspace":
        root_uri = Path.from_uri(doc.uri).absolute().parent.as_uri()
        return TestWorkspace(root_uri, [doc])

    def dump_references(self):
        rendered = []

        for key, ref_locations in self.index.def_to_refs.items():
            def_location = key.location
            def_doc = self.docs[def_location.uri]
            lhs = [Text.styled("Definition", "black on yellow")]
            lhs.append(def_doc.highlight([def_location.range], "black on blue"))

            rhs = [Text.styled("Reference(s)", "black on yellow")]
            for uri, group in groupby(ref_locations, key=lambda loc: loc.uri):
                ref_doc = self.docs[uri]
                ref_ranges = [loc.range for loc in group]
                rhs.append(ref_doc.highlight(ref_ranges, "black on blue"))

            rendered.append(side_by_side(Text("\n").join(lhs), Text("\n").join(rhs)))

        return Text("\n\n").join(rendered)

    def dump_definitions(self):
        rendered = []

        for key, ref_locations in self.index.ref_to_defs.items():
            ref_location = key.location
            ref_doc = self.docs[ref_location.uri]
            lhs = [Text.styled("Reference", "black on yellow")]
            lhs.append(ref_doc.highlight([ref_location.range], "black on green"))

            rhs = [Text.styled("Definition(s)", "black on yellow")]
            for uri, group in groupby(ref_locations, key=lambda loc: loc.uri):
                def_doc = self.docs[uri]
                def_ranges = [loc.range for loc in group]
                rhs.append(def_doc.highlight(def_ranges, "black on green"))

            rendered.append(side_by_side(Text("\n").join(lhs), Text("\n").join(rhs)))

        return Text("\n\n").join(rendered)


def side_by_side(lhs: Text | str, rhs: Text | str) -> Text:
    if isinstance(lhs, str):
        lhs = Text(lhs)

    if isinstance(rhs, str):
        rhs = Text(rhs)

    lhs_lines = lhs.split()
    rhs_lines = rhs.split()

    empty = Text.styled("", "default")
    shorter = lhs_lines if len(lhs_lines) < len(rhs_lines) else rhs_lines
    shorter.extend([empty] * abs(len(lhs_lines) - len(rhs_lines)))

    max_width = max(map(len, lhs.plain.splitlines()))
    sep = Text.styled(" : ", "grey50")

    return Text("\n").join(
        [
            lhs_line + padding + sep + rhs_line
            for lhs_line, rhs_line in zip(lhs_lines, rhs_lines)
            if (padding := " " * (max_width - len(lhs_line))) is not None
        ]
    )


class TestDefRef(unittest.TestCase):
    def assertLocationsEqual(
        self,
        workspace: TestWorkspace,
        obtained: list[L.Location],
        expected: list[L.Location],
        clue: Text | None = None,
    ):
        docs = workspace.docs.values()

        def sort_key(loc: L.Location):
            return (loc.uri, loc.range.start, loc.range.end)

        obtained.sort(key=sort_key)
        expected.sort(key=sort_key)

        def highlight(locations: list[L.Location], style: str) -> Text:
            return Text("\n" * 2).join(
                [
                    doc.highlight(ranges, style)
                    for uri, group in groupby(locations, lambda loc: loc.uri)
                    if (doc := next(d for d in docs if d.uri == uri)) is not None
                    if (ranges := list(location.range for location in group))
                ]
            )

        def header(title: str) -> Text:
            return Text.styled(title + "\n", "black on yellow")

        console = Console()

        with console.capture() as capture:
            from itertools import groupby

            console.print("\n")

            if clue is not None:
                console.print(clue)

            message = side_by_side(
                header("Expected") + highlight(expected, "black on green"),
                header("Obtained") + highlight(obtained, "black on red"),
            )

            console.print(message)

        self.assertSequenceEqual(obtained, expected, msg=capture.get())

    def checkDefRefs(
        self,
        workspace: TestWorkspace,
        def_location: L.Location,
        ref_locations: list[L.Location],
    ):
        def_doc = workspace.docs[def_location.uri]

        clue = Text("\n").join(
            [
                Text("Indexed definition(s) and reference(s):", style="default"),
                workspace.dump_references(),
                Text(""),
                workspace.dump_definitions(),
                Text(""),
                Text("Checking symbol reference(s):"),
                def_doc.highlight([def_location.range], "black on yellow"),
                Text(""),
            ]
        )

        self.assertLocationsEqual(
            workspace,
            workspace.index.references(
                def_location.uri,
                def_location.range.start,
            ),
            ref_locations,
            clue,
        )

        for location in ref_locations:
            ref_doc = workspace.docs[location.uri]

            clue = Text("\n").join(
                [
                    Text("Checking symbol definition:"),
                    ref_doc.highlight([location.range], "black on yellow"),
                    Text(""),
                    workspace.dump_definitions(),
                ]
            )

            self.assertLocationsEqual(
                workspace,
                workspace.index.definitions(
                    location.uri,
                    location.range.start,
                ),
                [def_location],
                clue,
            )

    def test_local_bind(self):
        t = TestDocument("local x = 1; x + x")

        self.checkDefRefs(
            TestWorkspace.single_doc(t),
            def_location=t.location_of("x"),
            ref_locations=[
                t.location_of("x", nth=2),
                t.location_of("x", nth=3),
            ],
        )

    def test_field(self):
        t = TestDocument("{ f: 1 }.f")

        self.checkDefRefs(
            TestWorkspace.single_doc(t),
            def_location=t.location_of("f: 1"),
            ref_locations=[t.location_of("f", nth=2)],
        )

    def test_nested_field(self):
        t = TestDocument("{ f: { g: 1 } }.f.g")

        self.checkDefRefs(
            TestWorkspace.single_doc(t),
            def_location=t.location_of("g: 1"),
            ref_locations=[t.location_of("g", nth=2)],
        )

    def test_obj_obj_composition(self):
        t = TestDocument(
            dedent(
                """\
                (
                    { f: 1 }
                    { f: 2 }
                ).f
                """
            )
        )

        self.checkDefRefs(
            TestWorkspace.single_doc(t),
            def_location=t.location_of("f: 2", line=3),
            ref_locations=[t.location_of("f", line=4)],
        )

    def test_var_obj_composition(self):
        t = TestDocument(
            dedent(
                """\
                local o = { f: 1 };
                (o + { f: 2 }).f
                """
            )
        )

        self.checkDefRefs(
            TestWorkspace.single_doc(t),
            def_location=t.location_of("f: 2", line=2),
            ref_locations=[t.location_of("f", line=2, nth=2)],
        )

    def test_obj_var_composition(self):
        t = TestDocument(
            dedent(
                """\
                local o = { f: 1 };
                ({ f: 2 } + o).f
                """
            )
        )

        self.checkDefRefs(
            TestWorkspace.single_doc(t),
            def_location=t.location_of("f: 1"),
            ref_locations=[t.location_of("f", line=2, nth=2)],
        )

    def test_var_var_composition(self):
        t = TestDocument(
            dedent(
                """\
                local o1 = { f: 1 };
                local o2 = { f: 2 };
                local o3 = { f: { g: 3 } };
                (o1 + o2 + o3).f.g
                """
            )
        )

        self.checkDefRefs(
            TestWorkspace.single_doc(t),
            def_location=t.location_of("g: 3", line=3),
            ref_locations=[t.location_of("g", line=4)],
        )

    def test_import(self):
        t1 = TestDocument(
            "{ f: 1 }",
            uri="file:///test/t1.jsonnet",
        )

        t2 = TestDocument(
            "(import 't1.jsonnet').f",
            uri="file:///test/t2.jsonnet",
        )

        self.checkDefRefs(
            workspace=TestWorkspace(
                root_uri="file:///test/",
                docs=[t1, t2],
            ),
            def_location=t1.location_of("f: 1"),
            ref_locations=[t2.location_of("f")],
        )

    def test_import_with_local(self):
        t1 = TestDocument(
            "local o1 = { f: 0 }; o1",
            uri="file:///test/t1.jsonnet",
        )

        t2 = TestDocument(
            "local o2 = import 't1.jsonnet'; o2.f",
            uri="file:///test/t2.jsonnet",
        )

        self.checkDefRefs(
            workspace=TestWorkspace(
                root_uri="file:///test/",
                docs=[t1, t2],
            ),
            def_location=t1.location_of("f: 0"),
            ref_locations=[t2.location_of("f")],
        )


if __name__ == "__main__":
    app()

# vim:ft=python tw=88
