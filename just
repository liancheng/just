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

import dataclasses as D
import logging
import sys
import unittest
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from textwrap import dedent
from typing import Annotated, Any, ClassVar, Optional, Type, TypeVar, cast

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

log = logging.root

LANG_JSONNET = get_language("jsonnet")


ASTType = TypeVar("ASTType")


def strip_comments(nodes: list[T.Node]) -> list[T.Node]:
    return [node for node in nodes if not node.type == "comment"]


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
            "document": Document.from_doc,
            "false": Bool.from_doc,
            "fieldname": FieldKey.from_doc,
            "field": Field.from_doc,
            "forloop": ListComp.from_doc,
            "forspec": ForSpec.from_doc,
            "functioncall": Call.from_doc,
            "id": Id.from_doc,
            "ifspec": IfSpec.from_doc,
            "import": Import.from_doc,
            "importstr": Import.from_doc,
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


def escape(s: str) -> str:
    escaped = s[0:20].translate(PrettyTree.ESCAPE_TABLE)
    postfix = "" if len(s) <= 10 else f"[{len(s) - 10} characters]"
    return f'"{escaped}{postfix}"'


@D.dataclass
class PrettyTree:
    value: Any
    label: str | None = None

    ESCAPE_TABLE: ClassVar[dict[int, str]] = str.maketrans(
        {"\n": r"\n", "\t": r"\t", "\r": r"\r", '"': r"\""}
    )

    @property
    def text(self):
        match self.value:
            case T.Node() as node if not node.is_named and node.text:
                repr = f"{escape(node.text.decode())} [{range_of(node)}]"
            case T.Node() as node if node.text:
                repr = f"{node.type} [{range_of(node)}]"
            case Document() as doc:
                repr = f"{doc.__class__.__name__} [{doc.location}]"
            case AST() as ast:
                repr = f"{ast.__class__.__name__} [{ast.location.range}]"
            case [_, *_]:
                repr = "[...]"
            case str():
                repr = escape(self.value)
            case _:
                repr = str(self.value)

        return repr if self.label is None else f"{self.label}={repr}"

    @property
    def children(self) -> list["PrettyTree"]:
        match self.value:
            case T.Node() as node:
                return [
                    PrettyTree(node.child(i), node.field_name_for_child(i))
                    for i in range(len(node.children))
                ]
            case AST() as ast:
                return [
                    PrettyTree(getattr(ast, f.name), f.name)
                    for f in D.fields(ast)
                    if f.name != "location"
                ]
            case [*elems] if (size := len(elems)) > 0:
                return [PrettyTree(elems[i], f"[{i}]") for i in range(size)]
            case _:
                return []

    def __repr__(self):
        def grow(buf: list[str], nodes: list[PrettyTree], branches: str = ""):
            for i in range(len(nodes)):
                last_child = i == len(nodes) - 1
                new_branch = ".   " if last_child else "|   "
                fork = "`-- " if last_child else "|-- "

                buf.append(f"{branches}{fork}{nodes[i].text}")
                grow(buf, nodes[i].children, branches + new_branch)

        lines = [self.text]
        grow(lines, self.children)

        return "\n".join(lines)


def pretty_tree(tree: AST | T.Node) -> str:
    return str(PrettyTree(tree))


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

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Id":
        assert node.type == "id"
        assert node.text is not None
        return Id(location_of(uri, node), node.text.decode())

    def bind(self, value: Expr) -> "Bind":
        return Bind(merge_locations(self, value), self, value)

    def arg(self, value: Expr) -> "Arg":
        return Arg(merge_locations(self, value), value, self)


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
        assert node.type == "binary"

        lhs, op, rhs = strip_comments(node.named_children)
        assert op.text is not None

        return Binary(
            location=location_of(uri, node),
            op=Operator(op.text.decode()),
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
        assert 2 <= len(children) <= 3

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

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Local":
        assert node.type == "local_bind"

        _, *binds, expr = strip_comments(node.named_children)

        return Local(
            location=location_of(uri, node),
            binds=[Bind.from_doc(uri, bind) for bind in binds],
            body=Expr.from_doc(uri, expr),
        )


@D.dataclass
class Param(AST):
    name: Id
    default: Expr | None = None

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Param":
        assert node.type == "param"

        children = strip_comments(node.named_children)
        assert 1 <= len(children) <= 2

        id, *maybe_value = children

        return Param(
            location=location_of(uri, node),
            name=Id.from_doc(uri, id),
            default=next(
                iter([Expr.from_doc(uri, value) for value in maybe_value]), None
            ),
        )


@D.dataclass
class Fn(Expr):
    params: list[Param]
    body: Expr

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Fn":
        assert node.type == "anonymous_function"

        children = strip_comments(node.named_children)
        assert 1 <= len(children) <= 2

        *maybe_params, body_node = children

        return Fn(
            location=location_of(uri, node),
            params=[
                Param.from_doc(uri, param)
                for params in maybe_params
                for param in strip_comments(params.named_children)
            ],
            body=Expr.from_doc(uri, body_node),
        )


@D.dataclass
class Arg(AST):
    value: Expr
    name: Id | None = None

    @staticmethod
    def from_doc(uri: str, node: T.Node) -> "Arg":
        if node.type == "named_argument":
            name, value = strip_comments(node.named_children)

            return Arg(
                location=location_of(uri, node),
                value=Expr.from_doc(uri, value),
                name=Id.from_doc(uri, name),
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
            case [child]:
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
        # However, `tree-sitter-jsonnet` explicitly hides the top node by prefixing the
        # rule with an underscore (`_assert_expr`), and therefore the parser only
        # returns the two child nodes:
        #
        #   <assert>          | assert true: "never"
        #       <_expr>       |        true
        #       <string>      |              "never"
        #   <id>              |                       x
        #
        # This complicates parsing values in local bindings and function bodies, because
        # a bound value and a function body can be either one node for a regular
        # expression or two nodes for an assertion expression.

        # Finds the first non-comment sibling node, which holds the body expression.
        body = node.next_named_sibling
        while body is not None and body.is_extra:
            body = body.next_named_sibling

        assert body is not None
        return AssertExpr(
            L.Location(uri, merge_ranges(node, body)),
            Assert.from_doc(uri, node),
            Expr.from_doc(uri, body),
        )


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
            return FixedKey(location, Id.from_doc(uri, head))
        else:
            return FixedKey(location, Str.from_doc(uri, head))


@D.dataclass
class FixedKey(FieldKey):
    name: Id | Str


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
        if p.default is not None:
            self.visit(p.default)

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


def location_of(uri: str, node: T.Node) -> L.Location:
    return L.Location(uri, range_of(node))


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


def range_contains(range: L.Range, pos: L.Position):
    return position_before(range.start, pos) and position_before(pos, range.end)


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

    assert position_before(start, end)

    return L.Range(start, end)


LocationLike = L.Location | AST


def merge_locations(lhs: LocationLike, rhs: LocationLike) -> L.Location:
    match lhs:
        case L.Location():
            start = lhs
        case AST():
            start = lhs.location

    match rhs:
        case L.Location():
            end = rhs
        case AST():
            end = rhs.location

    assert start.uri == end.uri, dedent(
        f"""\
        Cannot merge two locations from different documents:
        * {start.uri}
        * {end.uri}
        """
    )

    return L.Location(start.uri, merge_ranges(start.range, end.range))


def parse(source: str) -> T.Node:
    return get_parser("jsonnet").parse(source.encode()).root_node


@D.dataclass
class Binding:
    id: Id
    value: Expr


@D.dataclass
class Scope:
    bindings: list[Binding] = D.field(default_factory=list)
    parent: Optional["Scope"] = None

    def bind(self, id: Id, value: Expr):
        self.bindings.insert(0, Binding(id, value))

    def get(self, id: Id) -> Binding | None:
        maybe_binding = next(
            iter(b for b in self.bindings if b.id.name == id.name),
            None,
        )

        if maybe_binding is None and self.parent is not None:
            return self.parent.get(id)
        else:
            return maybe_binding

    def nest(self) -> "Scope":
        return Scope([], parent=self)


@D.dataclass
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


class Index(Visitor):
    def __init__(self, tree: Expr) -> None:
        self.root_symbol = L.DocumentSymbol(
            name="__root__",
            kind=L.SymbolKind.Variable,
            range=tree.location.range,
            selection_range=tree.location.range,
        )

        self.breadcrumbs = [self.root_symbol]
        self.scope = Scope()

        self.ref_to_defs: dict[HashableLocation, list[Id]] = {}
        self.def_to_refs: dict[HashableLocation, list[Id]] = {}

        self.visit(tree)

    def visit_bind(self, b: Bind):
        self.scope.bind(b.id, b.value)

        match b.value:
            case Fn():
                kind = L.SymbolKind.Function
            case _:
                kind = L.SymbolKind.Variable

        symbol = L.DocumentSymbol(
            name=b.id.name,
            kind=kind,
            range=b.id.location.range,
            selection_range=b.location.range,
        )

        with self.parent_symbol(symbol):
            with self.nested_scope():
                self.visit(b.value)

    def visit_param(self, p: Param):
        symbol = L.DocumentSymbol(
            name=p.name.name,
            kind=L.SymbolKind.Variable,
            range=p.name.location.range,
            selection_range=p.location.range,
        )

        with self.parent_symbol(symbol):
            if p.default is not None:
                self.visit(p.default)

    def visit_for_spec(self, s: ForSpec):
        self.visit(s.expr)

        symbol = L.DocumentSymbol(
            name=s.id.name,
            kind=L.SymbolKind.Variable,
            range=s.id.location.range,
            selection_range=s.id.location.range,
        )

        with self.parent_symbol(symbol):
            pass

    def visit_id(self, e: Id):
        if binding := self.scope.get(e):
            self.new_ref(e, binding)

    def new_ref(self, ref: Id, binding: Binding):
        defs = self.ref_to_defs.get(HashableLocation(ref.location), [])
        defs.append(binding.id)
        self.ref_to_defs[HashableLocation(ref.location)] = defs

        refs = self.def_to_refs.get(HashableLocation(binding.id.location), [])
        refs.append(ref)
        self.def_to_refs[HashableLocation(binding.id.location)] = refs

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

    @contextmanager
    def nested_scope(self):
        parent = self.scope
        self.scope = self.scope.nest()

        try:
            yield self.scope
        finally:
            self.scope = parent


class JustLanguageServer(LanguageServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.docs: dict[str, Index] = {}

    def document_root(self, uri: str) -> T.Node:
        return parse(self.workspace.get_text_document(uri).source)

    def refresh_index(self, uri: str) -> Index:
        index = Index(Expr.from_doc(uri, self.document_root(uri)))
        self.docs[uri] = index
        return index

    def index(self, uri: str) -> Index:
        if uri not in self.docs:
            self.refresh_index(uri)
        return self.docs[uri]


server = JustLanguageServer("just", "v0.1")


@server.feature(L.INITIALIZE)
def initialize(_: L.InitializeParams):
    return L.InitializeResult(
        capabilities=L.ServerCapabilities(
            document_symbol_provider=True,
            text_document_sync=L.TextDocumentSyncKind.Full,
        ),
        server_info=L.ServerInfo(
            name="just",
            version="v0.1",
        ),
    )


@server.feature(L.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: JustLanguageServer, params: L.DidOpenTextDocumentParams):
    ls.refresh_index(params.text_document.uri)


@server.feature(L.NOTEBOOK_DOCUMENT_DID_CHANGE)
def did_change(ls: JustLanguageServer, params: L.DidChangeTextDocumentParams):
    ls.refresh_index(params.text_document.uri)


@server.feature(L.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
def document_symbol(ls: JustLanguageServer, params: L.DocumentColorParams):
    uri = params.text_document.uri
    return ls.index(uri).root_symbol.children


@server.feature(L.TEXT_DOCUMENT_DEFINITION)
def definition(ls: JustLanguageServer, params: L.DefinitionParams):
    uri = params.text_document.uri
    return next(
        iter(
            [id.location for id in ids]
            for key, ids in ls.index(uri).ref_to_defs.items()
            if range_contains(key.location.range, params.position)
        ),
        [],
    )


@server.feature(L.TEXT_DOCUMENT_REFERENCES)
def references(ls: JustLanguageServer, params: L.ReferenceParams):
    uri = params.text_document.uri
    return next(
        iter(
            [id.location for id in ids]
            for key, ids in ls.index(uri).def_to_refs.items()
            if range_contains(key.location.range, params.position)
        ),
        [],
    )


class TestDocument:
    def __init__(self, source: str) -> None:
        self.uri = "file:///tmp/test.jsonnet"
        self.lines = source.splitlines(keepends=False)
        self.root = parse(source)
        self.body = Document.from_doc(self.uri, self.root).body

    def query_one(self, query: T.Query, capture: str) -> AST:
        [node] = T.QueryCursor(query).captures(self.root).get(capture, [])
        return AST.from_doc(self.uri, node)

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


class TestAST(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.maxDiff = None

    def assertAstEqual(self, tree_or_source: AST | str, expected: AST | str):
        match tree_or_source, expected:
            case AST() as tree, str():
                self.assertMultiLineEqual(pretty_tree(tree), expected.strip())
            case AST() as tree, AST() if tree != expected:
                self.assertAstEqual(tree, pretty_tree(expected))
            case str() as source, AST():
                self.assertAstEqual(TestDocument(source).body, pretty_tree(expected))
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
                |   .   |   `-- name="x"
                |   .   `-- value=Num [0:10-0:11]
                |   .   .   `-- value=1.0
                `-- body=Binary [0:13-0:18]
                .   |-- op="+"
                .   |-- lhs=Id [0:13-0:14]
                .   |   `-- name="x"
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

    def test_binary_precedences(self):
        t = TestDocument("a + b * c")

        self.assertAstEqual(
            t.body,
            t.id("a") + (t.id("b") * t.id("c")),
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
                t.id("x"),
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
                key=FixedKey(t.location_of("x"), t.id("x")),
                value=t.num(1),
            ),
        )

        t = TestDocument("{ x+::: 1 }")

        self.assertAstEqual(
            t.query_one(self.object_query, "field"),
            Field(
                location=t.location_of("x+::: 1"),
                key=FixedKey(t.location_of("x"), t.id("x")),
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
                key=FixedKey(t.location_of("f"), t.id("f")),
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
    ],
):
    if path == Path("-"):
        uri = "/dev/stdin"
        source = sys.stdin.read()
    else:
        uri = path.absolute().as_uri()
        source = path.read_text()

    cst = parse(source)
    ast = AST.from_doc(uri, cst)
    print(pretty_tree(cst if tree_sitter else ast))


@app.command()
def test():
    unittest.main(argv=["--locals"], verbosity=1)


if __name__ == "__main__":
    app()

# vim:ft=python tw=88
