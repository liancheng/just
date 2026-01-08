import unittest
from textwrap import dedent

import lsprotocol.types as L
import tree_sitter as T

from joule.ast import (
    AST,
    Array,
    Assert,
    AssertExpr,
    Binary,
    Call,
    DynamicKey,
    Field,
    FixedKey,
    Fn,
    ForSpec,
    IdKind,
    IfSpec,
    Import,
    ListComp,
    Local,
    ObjComp,
    Object,
    Operator,
    Str,
    Visibility,
    merge_locations,
)
from joule.parsing import LANG_JSONNET
from joule.util import head, maybe

from .dsl import FakeDocument


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
                self.assertAstEqual(FakeDocument(source).body, expected.pretty_tree)
            case str() as source, str():
                self.assertAstEqual(FakeDocument(source).body, expected)

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
                |   .   |   `-- kind="var"
                |   .   `-- value=Num [0:10-0:11]
                |   .   .   `-- value=1.0
                `-- body=Binary [0:13-0:18]
                .   |-- op="+"
                .   |-- lhs=Id [0:13-0:14]
                .   |   |-- name="x"
                .   |   `-- kind="varref"
                .   `-- rhs=Num [0:17-0:18]
                .   .   `-- value=2.0
                """
            ),
        )

    def test_number(self):
        t = FakeDocument("1")

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
            t = FakeDocument(literal)
            self.assertAstEqual(
                t.body,
                Str(t.body.location, expected),
            )

    def test_paren(self):
        t = FakeDocument("(1)")
        self.assertAstEqual(t.body, t.num(1))

        t = FakeDocument("(assert true; 1)")
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
        t = FakeDocument("local x = 1; x")

        self.assertAstEqual(
            t.body,
            Local(
                location=t.body.location,
                binds=[t.var("x").bind(t.num(1))],
                body=t.var_ref("x", nth=2),
            ),
        )

    def test_local_with_asserts(self):
        t = FakeDocument(
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
            body=t.var_ref("v1", line=4) + t.num(2, line=4),
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
                    t.var("v1").bind(t.num(0)),
                ],
                body=outer_assert_expr,
            ),
        )

    def test_local_bind_fn(self):
        t = FakeDocument(
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
                    t.var("f1", line=2).bind(
                        Fn(
                            location=t.location_of("f1(x) = x + 1", line=2),
                            params=[t.param("x", line=2)],
                            body=t.var_ref("x", line=2, nth=2)
                            + t.num(1, line=2, nth=2),
                        )
                    ),
                    t.var("f2", line=3).bind(
                        Fn(
                            location=t.location_of("f2(y, z) = y + z", line=3),
                            params=[
                                t.param("y", line=3),
                                t.param("z", line=3),
                            ],
                            body=t.var_ref("y", line=3, nth=2)
                            + t.var_ref("z", line=3, nth=2),
                        )
                    ),
                ],
                body=Call(
                    location=t.location_of("f2(f1(3), z = 4)", line=4),
                    fn=t.var_ref("f2", line=4),
                    args=[
                        t.arg(
                            Call(
                                location=t.location_of("f1(3)", line=4),
                                fn=t.var_ref("f1", line=4),
                                args=[t.arg(t.num(3, line=4))],
                            )
                        ),
                        t.arg(t.num(4, line=4), name="z", line=4),
                    ],
                ),
            ),
        )

    def test_local_multi_binds(self):
        t = FakeDocument(
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
                    t.var("x").bind(t.num(1)),
                    t.var("y").bind(t.num(2)),
                ],
                body=t.var_ref("x", line=2) + t.var_ref("y", line=2),
            ),
        )

    def test_empty_array(self):
        t = FakeDocument("[]")

        self.assertAstEqual(
            t.body,
            Array(location=t.body.location, values=[]),
        )

    def test_array(self):
        t = FakeDocument("[1, true, /* ! */ '3']")

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
            t = FakeDocument(f"a /*!*/ {op.value} /*!*/ b")

            self.assertAstEqual(
                t.body,
                t.var_ref("a").bin_op(op, t.var_ref("b")),
            )

    def test_implicit_plus(self):
        t = FakeDocument("a {}")

        self.assertAstEqual(
            t.body,
            t.var_ref("a") + Object(t.location_of("{}")),
        )

    def test_binary_precedences(self):
        t = FakeDocument("a + b * c")

        self.assertAstEqual(
            t.body,
            t.var_ref("a") + (t.var_ref("b") * t.var_ref("c")),
        )

        t = FakeDocument("(a + b) * c")

        self.assertAstEqual(
            t.body,
            Binary(
                t.body.location,
                op=Operator.Multiply,
                lhs=t.var_ref("a") + t.var_ref("b"),
                rhs=t.var_ref("c"),
            ),
        )

    def test_list_comp(self):
        t = FakeDocument("[x for x in [1, 2] if x > 1]")

        self.assertAstEqual(
            t.body,
            ListComp(
                location=t.body.location,
                expr=t.var_ref("x"),
                for_spec=ForSpec(
                    location=t.location_of("for x in [1, 2]"),
                    id=t.var("x", nth=2),
                    container=Array(
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
                        condition=t.var_ref("x", nth=3) > t.num(1, nth=2),
                    ),
                ],
            ),
        )

    def test_fn(self):
        t = FakeDocument("function(x, y = 2) x + y")

        self.assertAstEqual(
            t.body,
            Fn(
                t.body.location,
                params=[
                    t.param("x"),
                    t.param("y", default=t.num(2)),
                ],
                body=t.var_ref("x", nth=2) + t.var_ref("y", nth=2),
            ),
        )

    def test_fn_no_params(self):
        t = FakeDocument("function() 1")

        self.assertAstEqual(
            t.body,
            Fn(t.body.location, [], t.num(1)),
        )

    def test_import(self):
        t = FakeDocument("import 'test.jsonnet'")

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
        t = FakeDocument("importstr 'test.jsonnet'")

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
        t = FakeDocument("assert true; false")

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
        t = FakeDocument("assert true: 'never'; /*!*/ false")

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
        t = FakeDocument("local x = assert true; false; x")

        self.assertAstEqual(
            t.body,
            Local(
                location=t.body.location,
                binds=[
                    t.var("x").bind(
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
                body=t.var_ref("x", nth=2),
            ),
        )

    def test_nested_assert_expr(self):
        # Assertions are right associated.
        t = FakeDocument("assert true; assert false; x")

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
                    body=t.var_ref("x"),
                ),
            ),
        )

    def assertAstEqualByQuery(
        self, doc: FakeDocument, query: T.Query, capture: str, expected: AST
    ):
        captures = T.QueryCursor(query).captures(doc.root)
        [node] = captures[capture]
        self.assertAstEqual(AST.from_cst(doc.uri, node), expected)

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
        t = FakeDocument("local x = 'f'; { [x]: 1 }")

        self.assertAstEqual(
            t.query_one(self.object_query, "field_key"),
            DynamicKey(
                t.location_of("[x]"),
                t.var_ref("x", nth=2),
            ),
        )

        t = FakeDocument("{ x: 1 }")

        self.assertAstEqual(
            t.query_one(self.object_query, "field_key"),
            t.field("x"),
        )

        t = FakeDocument("{ 'x': 1 }")

        self.assertAstEqual(
            t.query_one(self.object_query, "field_key"),
            FixedKey(
                t.location_of("'x'"),
                t.str("x", literal="'x'"),
            ),
        )

    def test_field(self):
        t = FakeDocument("{ x: 1 }")

        self.assertAstEqual(
            t.query_one(self.object_query, "field"),
            Field(
                location=t.location_of("x: 1"),
                key=t.field("x"),
                value=t.num(1),
            ),
        )

    def test_hidden_field(self):
        t = FakeDocument("{ x+::: 1 }")

        self.assertAstEqual(
            t.query_one(self.object_query, "field"),
            Field(
                location=t.location_of("x+::: 1"),
                key=t.field("x"),
                value=t.num(1),
                visibility=Visibility.Forced,
                inherited=True,
            ),
        )

    def test_function_field(self):
        t = FakeDocument("{ f(p1, p2 = 0):: p1 + p2 }")

        self.assertAstEqual(
            t.query_one(self.object_query, "field"),
            Field(
                location=t.location_of("f(p1, p2 = 0):: p1 + p2"),
                key=t.field("f"),
                value=Fn(
                    t.location_of("f(p1, p2 = 0):: p1 + p2"),
                    params=[
                        t.param("p1"),
                        t.param("p2", default=t.num(0)),
                    ],
                    body=t.var_ref("p1", nth=2) + t.var_ref("p2", nth=2),
                ),
                visibility=Visibility.Hidden,
            ),
        )

    def test_function_field_no_params(self):
        t = FakeDocument("{ f():: 1 }")

        self.assertAstEqual(
            t.query_one(self.object_query, "field"),
            Field(
                location=t.location_of("f():: 1"),
                key=t.field("f"),
                value=Fn(
                    t.location_of("f():: 1"),
                    params=[],
                    body=t.num(1),
                ),
                visibility=Visibility.Hidden,
            ),
        )

    def test_obj_comp(self):
        t = FakeDocument(
            dedent(
                """\
                {
                    ['f' + x]: 0
                    for x in [1, 2]
                    if x > 1
                }
                """
            )
        )

        self.assertAstEqual(
            t.query_one(self.object_query, "object"),
            ObjComp(
                location=t.body.location,
                binds=[],
                field=Field(
                    location=t.location_of("['f' + x]: 0", line=2),
                    key=DynamicKey(
                        t.location_of("['f' + x]", line=2),
                        t.str("f", literal="'f'", line=2) + t.var_ref("x", line=2),
                    ),
                    value=t.num(0, line=2),
                ),
                for_spec=ForSpec(
                    location=t.location_of("for x in [1, 2]", line=3),
                    id=t.var("x", line=3),
                    container=Array(
                        location=t.location_of("[1, 2]", line=3),
                        values=[t.num(1, line=3), t.num(2, line=3)],
                    ),
                ),
                comp_spec=[
                    IfSpec(
                        location=t.location_of("if x > 1", line=4),
                        condition=t.var_ref("x", line=4) > t.num(1, line=4),
                    )
                ],
            ),
        )

    @unittest.skip("tree-sitter-jsonnet does not handle `objforloop` with local.")
    def test_obj_comp_with_local(self):
        FakeDocument(
            dedent(
                """\
                {
                    local y = 3,
                    ['f' + x]: x,
                    local z = 4,
                    for x in [1, 2]
                    if x > 1
                }
                """
            )
        )

    def test_narrowest_enclosing_node(self):
        t = FakeDocument("{ f: local x = 1; x }")

        def find_narrowest_node(position: L.Position) -> AST:
            return head(maybe(t.body.narrowest_node(position)))

        self.assertAstEqual(
            find_narrowest_node(t.start_of("x")),
            t.id("x", IdKind.Var),
        )

        self.assertAstEqual(
            find_narrowest_node(t.start_of("=")),
            t.id("x", IdKind.Var).bind(t.num(1)),
        )

        self.assertAstEqual(
            find_narrowest_node(t.start_of("local")),
            Local(
                t.location_of("local x = 1; x"),
                [t.id("x", IdKind.Var).bind(t.num(1))],
                t.id("x", IdKind.VarRef, nth=2),
            ),
        )

        local = Local(
            t.location_of("local x = 1; x"),
            binds=[t.id("x", IdKind.Var).bind(t.num(1))],
            body=t.id("x", IdKind.VarRef, nth=2),
        )

        self.assertAstEqual(
            find_narrowest_node(t.end_of(";")),
            local,
        )

        self.assertAstEqual(
            find_narrowest_node(t.end_of(":")),
            Field(
                t.location_of("f: local x = 1; x"),
                key=FixedKey(t.location_of("f"), t.id("f", IdKind.Field)),
                value=local,
            ),
        )
