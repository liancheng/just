import dataclasses as D
from enum import StrEnum, auto
from itertools import dropwhile
from textwrap import dedent
from typing import Any, Callable, ClassVar, Iterator, cast

import lsprotocol.types as L
import tree_sitter as T

from joule.pretty import PrettyTree
from joule.typing import URI
from joule.util import head_or_none, maybe


def strip_comments(nodes: list[T.Node]) -> list[T.Node]:
    return [node for node in nodes if not node.type == "comment"]


class IdKind(StrEnum):
    Var = auto()
    VarRef = auto()
    Field = auto()
    FieldRef = auto()
    Param = auto()
    CallArg = auto()


ParseCST = Callable[[str, T.Node], "AST"]


@D.dataclass
class AST:
    location: L.Location

    registry: ClassVar[dict[str, ParseCST]] = {}

    @staticmethod
    def register(fn: ParseCST, *node_types: str):
        for node_type in node_types:
            AST.registry[node_type] = fn

    @staticmethod
    def skip_paren(uri: URI, node: T.Node):
        return AST.from_cst(uri, strip_comments(node.named_children)[0])

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> AST:
        try:
            cst_parser = AST.registry.get(node.type, Unknown.from_cst)
            return cst_parser(uri, node)
        except Exception:
            return Error.from_cst(uri, node)

    @property
    def pretty_tree(self) -> str:
        return str(PrettyAST(self))


def skip_parenthesis(uri: URI, node: T.Node) -> Expr:
    return Expr.from_cst(uri, strip_comments(node.named_children)[0])


AST.register(skip_parenthesis, "parenthesis")


ESCAPE_TABLE: dict[int, str] = str.maketrans(
    {"\n": r"\n", "\t": r"\t", "\r": r"\r", '"': r"\""}
)


def escape(s: str, size: int = 50) -> str:
    escaped = s[0:size].translate(ESCAPE_TABLE)
    postfix = "" if len(s) <= size else f"[{len(s) - size} characters]"
    return f'"{escaped}{postfix}"'


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
class Expr(AST):
    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Expr:
        if isinstance(ast := AST.from_cst(uri, node), Expr):
            return cast(Expr, ast)
        else:
            raise TypeError(f"Expected {Expr.__name__}, but got {type(ast).__name__}")

    @property
    def tails(self) -> list[Expr]:
        return [self]

    def bin_op(self, op: "Operator", rhs: "Expr") -> Binary:
        return Binary(merge_locations(self, rhs), op, self, rhs)

    def __add__(self, rhs: "Expr") -> Binary:
        return self.bin_op(Operator.Plus, rhs)

    def __sub__(self, rhs: "Expr") -> Binary:
        return self.bin_op(Operator.Minus, rhs)

    def __mul__(self, rhs: "Expr") -> Binary:
        return self.bin_op(Operator.Multiply, rhs)

    def __truediv__(self, rhs: "Expr") -> Binary:
        return self.bin_op(Operator.Divide, rhs)

    def __lt__(self, rhs: "Expr") -> Binary:
        return self.bin_op(Operator.LT, rhs)

    def __le__(self, rhs: "Expr") -> Binary:
        return self.bin_op(Operator.LE, rhs)

    def __gt__(self, rhs: "Expr") -> Binary:
        return self.bin_op(Operator.GT, rhs)

    def __ge__(self, rhs: "Expr") -> Binary:
        return self.bin_op(Operator.GE, rhs)

    def eq(self, rhs: object) -> Binary:
        assert isinstance(rhs, Expr)
        return self.bin_op(Operator.Eq, rhs)

    def not_eq(self, rhs: "Expr") -> Binary:
        return self.bin_op(Operator.NotEq, rhs)


@D.dataclass
class Self(Expr):
    def __post_init__(self):
        self.scope: Scope | None = None

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Self:
        assert node.type == "self"
        return Self(location_of(uri, node))

    AST.register(from_cst, "self")


@D.dataclass
class Super(Expr):
    def __post_init__(self):
        self.scope: Scope | None = None

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Super:
        assert node.type == "super"
        return Super(location_of(uri, node))

    AST.register(from_cst, "super")


@D.dataclass
class Document(Expr):
    body: Expr

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Document:
        assert node.type == "document"
        body, *_ = strip_comments(node.named_children)
        return Document(location_of(uri, node), Expr.from_cst(uri, body))

    @property
    def tails(self) -> list[Expr]:
        return self.body.tails

    AST.register(from_cst, "document")


@D.dataclass
class Id(Expr):
    name: str
    kind: IdKind

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Id:
        assert node.type == "id"
        assert node.text is not None

        # By default, an ID is a variable reference. IDs of other kinds are always
        # parsed while parsing other specific AST nodes, where the `IdKind` is
        # explicitly specified.
        return Id(location_of(uri, node), node.text.decode(), IdKind.VarRef)

    def bind(self, value: Expr) -> Bind:
        return Bind(merge_locations(self, value), self, value)

    def arg(self, value: Expr) -> Arg:
        return Arg(merge_locations(self, value), value, self.into(IdKind.CallArg))

    def into(self, kind: IdKind) -> Id:
        self.kind = kind
        return self

    @property
    def is_variable(self) -> bool:
        return self.kind in [IdKind.VarRef, IdKind.FieldRef]

    AST.register(from_cst, "id")


@D.dataclass
class Num(Expr):
    value: float

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Num:
        assert node.type == "number"
        assert node.text is not None
        return Num(location_of(uri, node), float(node.text.decode()))

    AST.register(from_cst, "number")


@D.dataclass
class Str(Expr):
    raw: str

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Str:
        assert node.type == "string"

        _, content, _ = node.named_children
        assert content.text is not None

        return Str(
            location=location_of(uri, node),
            # The raw string content before escaping or indentations are handled.
            raw=content.text.decode(),
        )

    AST.register(from_cst, "string")


@D.dataclass
class Bool(Expr):
    value: bool

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Bool:
        assert node.type in ["true", "false"]
        assert node.text is not None
        return Bool(location_of(uri, node), node.text.decode() == "true")

    AST.register(from_cst, "false", "true")


@D.dataclass
class Array(Expr):
    values: list[Expr]

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Array:
        assert node.type == "array"
        return Array(
            location=location_of(uri, node),
            values=[
                Expr.from_cst(uri, child)
                for child in strip_comments(node.named_children)
            ],
        )

    AST.register(from_cst, "array")


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
    def from_cst(uri: URI, node: T.Node) -> Binary:
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
            lhs=Expr.from_cst(uri, lhs),
            rhs=Expr.from_cst(uri, rhs),
        )

    AST.register(from_cst, "binary", "implicit_plus")


@D.dataclass
class Bind(AST):
    id: Id
    value: Expr

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Bind:
        assert node.type == "bind"

        children = strip_comments(node.named_children)

        if (fn_name := node.child_by_field_name("function")) is not None:
            maybe_params = maybe(node.child_by_field_name("params"))
            first_body, *_ = strip_comments(node.children_by_field_name("body"))

            fn = Fn(
                location=location_of(uri, node),
                params=[
                    Param.from_cst(uri, param)
                    for params in maybe_params
                    for param in strip_comments(params.named_children)
                ],
                body=Expr.from_cst(uri, first_body),
            )

            return Bind(
                location=fn.location,
                id=Id.from_cst(uri, fn_name).into(IdKind.Var),
                value=fn,
            )
        else:
            id, value, *_ = children
            return Bind(
                location=location_of(uri, node),
                id=Id.from_cst(uri, id).into(IdKind.Var),
                value=Expr.from_cst(uri, value),
            )

    AST.register(from_cst, "bind")


@D.dataclass
class Local(Expr):
    binds: list[Bind]
    body: Expr

    def __post_init__(self):
        self.scope: Scope | None = None

    @property
    def tails(self) -> list[Expr]:
        return self.body.tails

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Local:
        assert node.type == "local_bind"

        # Skips the first node, which is the "local" keyword.
        _, *children = strip_comments(node.named_children)

        binds, body = [], []
        for child in children:
            (binds if child.type == "bind" else body).append(child)

        return Local(
            location=location_of(uri, node),
            binds=[Bind.from_cst(uri, bind) for bind in binds],
            body=Expr.from_cst(uri, body[0]),
        )

    AST.register(from_cst, "local_bind")


@D.dataclass
class Param(AST):
    id: Id
    default: Expr | None = None

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Param:
        assert node.type == "param"

        children = strip_comments(node.named_children)
        assert 1 <= len(children) <= 2

        id, *maybe_default = children
        return Param(
            location=location_of(uri, node),
            id=Id.from_cst(uri, id).into(IdKind.Param),
            default=head_or_none(Expr.from_cst(uri, value) for value in maybe_default),
        )

    AST.register(from_cst, "param")


@D.dataclass
class Fn(Expr):
    params: list[Param]
    body: Expr

    @property
    def tails(self) -> list[Expr]:
        return self.body.tails

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Fn:
        assert node.type == "anonymous_function"
        first, *rest = strip_comments(node.named_children)

        match first, rest:
            case _, (body, *_) if first.type == "params":
                params = [
                    Param.from_cst(uri, param)
                    for param in strip_comments(first.named_children)
                ]
            case body, *_:
                params = []

        return Fn(
            location=location_of(uri, node),
            params=params,
            body=Expr.from_cst(uri, body),
        )

    AST.register(from_cst, "anonymous_function")


@D.dataclass
class Arg(AST):
    value: Expr
    id: Id | None = None

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Arg:
        if node.type == "named_argument":
            name, value = strip_comments(node.named_children)

            return Arg(
                location=location_of(uri, node),
                value=Expr.from_cst(uri, value),
                id=Id.from_cst(uri, name).into(IdKind.CallArg),
            )
        else:
            return Arg(
                location=location_of(uri, node),
                value=Expr.from_cst(uri, node),
            )

    AST.register(from_cst, "named_argument")


@D.dataclass
class Call(Expr):
    fn: Expr
    args: list[Arg]

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Call:
        assert node.type == "functioncall"

        children = strip_comments(node.named_children)
        assert len(children) >= 1

        fn, *maybe_args = children

        return Call(
            location=location_of(uri, node),
            fn=Expr.from_cst(uri, fn),
            args=[
                Arg.from_cst(uri, arg)
                for args in maybe_args
                for arg in strip_comments(args.named_children)
            ],
        )

    AST.register(from_cst, "functioncall")


@D.dataclass
class ForSpec(AST):
    id: Id
    expr: Expr

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> ForSpec:
        assert node.type == "forspec"
        id, expr = strip_comments(node.named_children)

        return ForSpec(
            location_of(uri, node),
            Id.from_cst(uri, id).into(IdKind.Var),
            Expr.from_cst(uri, expr),
        )

    AST.register(from_cst, "forspec")


@D.dataclass
class IfSpec(AST):
    condition: Expr

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> IfSpec:
        assert node.type == "ifspec"
        [child] = strip_comments(node.named_children)
        return IfSpec(location_of(uri, node), condition=Expr.from_cst(uri, child))

    AST.register(from_cst, "ifspec")


@D.dataclass
class ListComp(Expr):
    expr: Expr
    for_spec: ForSpec
    comp_spec: list[ForSpec | IfSpec]

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> ListComp:
        assert node.type == "forloop"

        children = strip_comments(node.named_children)
        assert len(children) >= 2

        expr, for_spec, *maybe_comp_spec = children
        return ListComp(
            location=location_of(uri, node),
            expr=Expr.from_cst(uri, expr),
            for_spec=ForSpec.from_cst(uri, for_spec),
            comp_spec=[
                ForSpec.from_cst(uri, spec)
                if spec.type == "forspec"
                else IfSpec.from_cst(uri, spec)
                for comp_spec in maybe_comp_spec
                for spec in strip_comments(comp_spec.named_children)
            ],
        )

    AST.register(from_cst, "forloop")


@D.dataclass
class Import(Expr):
    type: str
    path: Str

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Import:
        assert node.type in ["import", "importstr"]

        [path] = strip_comments(node.named_children)
        return Import(
            location_of(uri, node),
            node.type,
            Str.from_cst(uri, path),
        )

    AST.register(from_cst, "import", "importstr")


@D.dataclass
class Assert(AST):
    condition: Expr
    message: Expr | None = None

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Assert:
        assert node.type == "assert"

        children = strip_comments(node.named_children)
        assert 1 <= len(children) <= 2

        condition, *maybe_message = children

        match maybe_message:
            case child, *_:
                message = Expr.from_cst(uri, child)
            case _:
                message = None

        return Assert(
            location_of(uri, node),
            condition=Expr.from_cst(uri, condition),
            message=message,
        )


@D.dataclass
class AssertExpr(Expr):
    assertion: Assert
    body: Expr

    @property
    def tails(self) -> list[Expr]:
        return self.body.tails

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Expr:
        # An `AssertExpr` is an expression consisting of an assertion followed by an
        # expression. Ideally, one expression should map to one `Expr` node and one
        # tree-sitter node. However, `AssertExpr` breaks this nice property.
        #
        # Consider the following assert expression:
        #
        #   assert true;
        #   assert true;
        #   x + 1
        #
        # If tree-sitter returns the following tree, it would be straightforward to
        # parse:
        #
        #   <assert_expr>
        #       <assert>
        #           <true>
        #       <assert_expr>
        #           <assert>
        #               <true>
        #           <binary>
        #               left: <id>
        #               operator: <additive>
        #               right: <id>
        #
        # However, `tree-sitter-jsonnet` explicitly hides the top node by prefixing the
        # rule name with an underscore (`_assert_expr` instead of `assert_expr`), which
        # forces the parser to return a forest of trees:
        #
        #   <assert>
        #       <true>
        #   <assert>
        #       <true>
        #   <binary>
        #       left: <id>
        #       operator: <additive>
        #       right: <id>
        #
        # This unnecessarily complicates the parsing logic, because an `Expr` node may
        # map to one or more tree-sitter nodes.

        def named_siblings(node: T.Node) -> Iterator[T.Node]:
            sibling = node.next_named_sibling
            while sibling is not None:
                yield sibling
                sibling = sibling.next_named_sibling

        # Finds the first non-comment sibling named node. This is the first node of the
        # body expression.
        no_comments = dropwhile(
            lambda sibling: sibling.is_extra,
            named_siblings(node),
        )

        assertion = Assert.from_cst(uri, node)
        body = Expr.from_cst(uri, next(no_comments))

        return AssertExpr(merge_locations(assertion, body), assertion, body)

    AST.register(from_cst, "assert")


@D.dataclass
class FieldKey(AST):
    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> FieldKey:
        assert node.type == "fieldname"

        location = location_of(uri, node)
        head, *tail = strip_comments(node.children)
        assert head.text is not None

        if head.text.decode() == "[":
            e, *_ = tail
            return DynamicKey(location, Expr.from_cst(uri, e))
        elif head.type == "id":
            return FixedKey(location, Id.from_cst(uri, head).into(IdKind.Field))
        else:
            return FixedKey(location, Str.from_cst(uri, head))

    AST.register(from_cst, "fieldname")


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
        self.enclosing_obj: Object | None = None

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Field:
        assert node.type == "field"
        children = strip_comments(node.children)

        def parse_function_field(children: list[T.Node]) -> Field:
            key, _, params_or_paren, *rest = children

            match params_or_paren:
                case params if params.type == "params":
                    _, vis, body, *_ = rest
                    params = [
                        Param.from_cst(uri, param)
                        for param in strip_comments(params.named_children)
                    ]
                case _:
                    params = []
                    vis, body, *_ = rest

            assert vis.text is not None
            return Field(
                location=location_of(uri, node),
                key=FieldKey.from_cst(uri, key),
                value=Fn(
                    location_of(uri, node),
                    params=params,
                    body=Expr.from_cst(uri, body),
                ),
                visibility=Visibility(vis.text.decode()),
            )

        def parse_value_field(children: list[T.Node]) -> Field:
            key, plus_or_vis, *rest = children

            match plus_or_vis:
                case plus if plus.text == b"+":
                    vis, value, *_ = rest
                    assert vis.text is not None
                    inherited = True
                case vis:
                    assert vis.text is not None
                    value, *_ = rest
                    inherited = False

            return Field(
                location=location_of(uri, node),
                key=FieldKey.from_cst(uri, key),
                value=Expr.from_cst(uri, value),
                visibility=Visibility(vis.text.decode()),
                inherited=inherited,
            )

        return (
            parse_function_field(children)
            if len(node.children_by_field_name("function")) > 0
            else parse_value_field(children)
        )

    AST.register(from_cst, "field")


@D.dataclass
class Object(Expr):
    binds: list[Bind] = D.field(default_factory=list)
    assertions: list[Assert] = D.field(default_factory=list)
    fields: list[Field] = D.field(default_factory=list)

    def __post_init__(self):
        self.var_scope: Scope | None = None
        self.super_scope: Scope | None = None
        self.self_scope: Scope | None = None

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Object | ObjComp:
        assert node.type == "object"

        match strip_comments(node.named_children):
            case head, *_ if head.type == "objforloop":
                return ObjComp.from_cst(uri, head)
            case _:
                binds = []
                assertions = []
                fields = []

                for member in strip_comments(node.named_children):
                    assert member.type == "member"
                    head, *_ = strip_comments(member.named_children)
                    match head.type:
                        case "objlocal":
                            _, bind, *_ = strip_comments(head.named_children)
                            binds.append(Bind.from_cst(uri, bind))
                        case "assert":
                            assertions.append(Assert.from_cst(uri, head))
                        case "field":
                            fields.append(Field.from_cst(uri, head))

                return Object(location_of(uri, node), binds, assertions, fields)

    AST.register(from_cst, "object")


@D.dataclass
class ObjComp(Expr):
    binds: list[Bind]
    field: Field
    for_spec: ForSpec
    comp_spec: list[ForSpec | IfSpec] = D.field(default_factory=list)

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> ObjComp:
        assert node.type == "objforloop"
        field, for_spec, *maybe_comp_spec = node.named_children

        return ObjComp(
            location=location_of(uri, node),
            binds=[],
            field=Field.from_cst(uri, field),
            for_spec=ForSpec.from_cst(uri, for_spec),
            comp_spec=[
                ForSpec.from_cst(uri, spec)
                if spec.type == "forspec"
                else IfSpec.from_cst(uri, spec)
                for comp_spec in maybe_comp_spec
                for spec in strip_comments(comp_spec.named_children)
            ],
        )


@D.dataclass
class FieldAccess(Expr):
    obj: Expr
    field: Id

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> FieldAccess:
        assert node.type in ["fieldaccess", "fieldaccess_super"]
        expr, field = strip_comments(node.named_children)
        return FieldAccess(
            location=location_of(uri, node),
            obj=Expr.from_cst(uri, expr),
            field=Id.from_cst(uri, field).into(IdKind.FieldRef),
        )

    AST.register(from_cst, "fieldaccess_super", "fieldaccess")


@D.dataclass
class Slice(Expr):
    array: Expr
    begin: Expr
    end: Expr | None = None
    step: Expr | None = None

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Slice:
        assert node.type == "indexing"
        expr, begin, *rest = strip_comments(node.named_children)
        match rest:
            case [end, step]:
                end = Expr.from_cst(uri, end)
                step = Expr.from_cst(uri, step)
            case [end]:
                end = Expr.from_cst(uri, end)
                step = None
            case _:
                end = None
                step = None

        return Slice(
            location=location_of(uri, node),
            array=Expr.from_cst(uri, expr),
            begin=Expr.from_cst(uri, begin),
            end=end,
            step=step,
        )

    AST.register(from_cst, "indexing")


@D.dataclass
class If(Expr):
    condition: Expr
    consequence: Expr
    alternative: Expr | None

    @property
    def tails(self) -> list[Expr]:
        return self.consequence.tails + [
            tail
            for alternative in maybe(self.alternative)
            for tail in alternative.tails
        ]

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> If:
        assert node.type == "conditional"
        condition, consequence, *maybe_alternative = strip_comments(node.named_children)
        return If(
            location=location_of(uri, node),
            condition=Expr.from_cst(uri, condition),
            consequence=Expr.from_cst(uri, consequence),
            alternative=head_or_none(
                Expr.from_cst(uri, alternative) for alternative in maybe_alternative
            ),
        )

    AST.register(from_cst, "conditional")


@D.dataclass
class Unknown(Expr):
    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Unknown:
        return Unknown(location_of(uri, node))


@D.dataclass
class Error(Expr):
    node_type: str

    @staticmethod
    def from_cst(uri: URI, node: T.Node) -> Error:
        return Error(location_of(uri, node), node.type)


def position_of(point: T.Point) -> L.Position:
    return L.Position(point.row, point.column)


def range_of(node: T.Node) -> L.Range:
    return L.Range(
        position_of(node.range.start_point),
        position_of(node.range.end_point),
    )


def location_of(uri: URI, node: T.Node) -> L.Location:
    return L.Location(uri, range_of(node))


RangeLike = L.Range | T.Range | T.Node | AST


def merge_ranges(lhs: RangeLike, rhs: RangeLike) -> L.Range:
    match lhs:
        case L.Range():
            start = lhs.start
        case T.Range() as r:
            start = position_of(r.start_point)
        case T.Node() as n:
            start = position_of(n.start_point)
        case AST():
            start = lhs.location.range.start

    match rhs:
        case L.Range():
            end = rhs.end
        case T.Range() as r:
            end = position_of(r.end_point)
        case T.Node() as n:
            end = position_of(n.end_point)
        case AST():
            end = rhs.location.range.end

    assert start < end or end == start

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


@D.dataclass
class Binding:
    scope: "Scope"
    name: str
    location: L.Location
    target: AST | None = None


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

    def nest(self) -> Scope:
        child = Scope([], parent=self)
        self.children.append(child)
        return child

    @property
    def pretty_tree(self) -> str:
        return str(PrettyScope(self))

    @staticmethod
    def empty() -> Scope:
        return Scope()


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
