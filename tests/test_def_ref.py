import unittest
from textwrap import dedent

import lsprotocol.types as L
from rich.console import Console
from rich.text import Text

from .dsl import FakeDocument, FakeWorkspace, side_by_side

TITLE_STYLE = "black on yellow"
DEFAULT_HIGHLIGHT_STYLE = "black on blue"
EXPECTED_HIGHLIGHT_STYLE = "black on green"
OBSERVED_HIGHLIGHT_STYLE = "black on red"


class TestDefRef(unittest.TestCase):
    def assertLocationsEqual(
        self,
        workspace: FakeWorkspace,
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
            return Text.styled(title + "\n", TITLE_STYLE)

        console = Console()

        with console.capture() as capture:
            from itertools import groupby

            console.print("\n")

            if clue is not None:
                console.print(clue)

            message = side_by_side(
                header("Expected") + highlight(expected, EXPECTED_HIGHLIGHT_STYLE),
                header("Obtained") + highlight(obtained, OBSERVED_HIGHLIGHT_STYLE),
            )

            console.print(message)

        self.assertSequenceEqual(obtained, expected, msg=capture.get())

    def checkDefRefs(
        self,
        workspace: FakeWorkspace,
        def_location: L.Location,
        ref_locations: list[L.Location],
    ):
        def_doc = workspace.docs[def_location.uri]

        clue = Text("\n").join(
            [
                Text("Checking symbol reference(s):"),
                def_doc.highlight([def_location.range], TITLE_STYLE),
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
                    ref_doc.highlight([location.range], TITLE_STYLE),
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
        t = FakeDocument("local x = 1; x + x")

        self.checkDefRefs(
            FakeWorkspace.single_doc(t),
            def_location=t.location_of("x"),
            ref_locations=[
                t.location_of("x", nth=2),
                t.location_of("x", nth=3),
            ],
        )

    def test_field(self):
        t = FakeDocument("{ f: 1 }.f")

        self.checkDefRefs(
            FakeWorkspace.single_doc(t),
            def_location=t.location_of("f"),
            ref_locations=[t.location_of("f", nth=2)],
        )

    def test_nested_field(self):
        t = FakeDocument("{ f: { g: 1 } }.f.g")

        self.checkDefRefs(
            FakeWorkspace.single_doc(t),
            def_location=t.location_of("g"),
            ref_locations=[t.location_of("g", nth=2)],
        )

    def test_obj_obj_composition(self):
        t = FakeDocument(
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
            FakeWorkspace.single_doc(t),
            def_location=t.location_of("f", line=3),
            ref_locations=[t.location_of("f", line=4)],
        )

    def test_var_obj_composition(self):
        t = FakeDocument(
            dedent(
                """\
                local o = { f: 1 };
                (o + { f: 2 }).f
                """
            )
        )

        self.checkDefRefs(
            FakeWorkspace.single_doc(t),
            def_location=t.location_of("f", line=2),
            ref_locations=[t.location_of("f", line=2, nth=2)],
        )

    def test_obj_var_composition(self):
        t = FakeDocument(
            dedent(
                """\
                local o = { f: 1 };
                ({ f: 2 } + o).f
                """
            )
        )

        self.checkDefRefs(
            FakeWorkspace.single_doc(t),
            def_location=t.location_of("f"),
            ref_locations=[t.location_of("f", line=2, nth=2)],
        )

    def test_var_var_composition(self):
        t = FakeDocument(
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
            FakeWorkspace.single_doc(t),
            def_location=t.location_of("g", line=3),
            ref_locations=[t.location_of("g", line=4)],
        )

    @unittest.skip("Cross-document references not implemented")
    def test_import(self):
        t1 = FakeDocument(
            "{ f: 1 }",
            uri="file:///test/t1.jsonnet",
        )

        t2 = FakeDocument(
            "(import 't1.jsonnet').f",
            uri="file:///test/t2.jsonnet",
        )

        self.checkDefRefs(
            FakeWorkspace(
                root_uri="file:///test/",
                docs=[t1, t2],
            ),
            def_location=t1.location_of("f: 1"),
            ref_locations=[t2.location_of("f")],
        )

    def test_same_var_field_names(self):
        t1 = FakeDocument(
            dedent(
                """\
                local f = { f: 1 };
                {
                    f: f,
                }.f.f
                """
            ),
            uri="file:///test/t1.jsonnet",
        )

        FakeWorkspace.single_doc(t1)

        self.checkDefRefs(
            FakeWorkspace.single_doc(t1),
            def_location=t1.location_of("f", line=1),
            ref_locations=[t1.location_of("f", line=3, nth=2)],
        )

        self.checkDefRefs(
            FakeWorkspace.single_doc(t1),
            def_location=t1.location_of("f", line=3, nth=1),
            ref_locations=[t1.location_of("f", line=4, nth=1)],
        )

        self.checkDefRefs(
            FakeWorkspace.single_doc(t1),
            def_location=t1.location_of("f", line=1, nth=2),
            ref_locations=[t1.location_of("f", line=4, nth=2)],
        )

    @unittest.skip("Cross-document references not implemented")
    def test_import_with_local(self):
        t1 = FakeDocument(
            "local o1 = { f: 0 }; o1",
            uri="file:///test/t1.jsonnet",
        )

        t2 = FakeDocument(
            "local o2 = import 't1.jsonnet'; o2.f",
            uri="file:///test/t2.jsonnet",
        )

        self.checkDefRefs(
            FakeWorkspace(
                root_uri="file:///test/",
                docs=[t1, t2],
            ),
            def_location=t1.location_of("f: 0"),
            ref_locations=[t2.location_of("f")],
        )

    def test_self(self):
        t = FakeDocument("{ f1: 1, f2: self.f1 }")

        self.checkDefRefs(
            FakeWorkspace.single_doc(t),
            def_location=t.location_of("f1"),
            ref_locations=[t.location_of("f1", nth=2)],
        )

    def test_super_obj_obj(self):
        t = FakeDocument(
            dedent(
                """\
                {
                    f1: 1
                } + {
                    f1: 2,
                    f2: super.f1
                }
                """
            )
        )

        self.checkDefRefs(
            FakeWorkspace.single_doc(t),
            def_location=t.location_of("f1", line=2),
            ref_locations=[t.location_of("f1", line=5)],
        )

    def test_super_var_obj(self):
        t = FakeDocument(
            dedent(
                """\
                local o1 = {
                    f1: 1
                };
                o1 + {
                    f1: 2,
                    f2: super.f1
                }
                """
            )
        )

        self.checkDefRefs(
            FakeWorkspace.single_doc(t),
            def_location=t.location_of("f1", line=2),
            ref_locations=[t.location_of("f1", line=6)],
        )

        self.checkDefRefs(
            FakeWorkspace.single_doc(t),
            def_location=t.location_of("o1"),
            ref_locations=[t.location_of("o1", line=4)],
        )

    def test_super_var_var(self):
        t = FakeDocument(
            dedent(
                """\
                /* 1 */ local o1 = {
                /* 2 */     f1: 1
                /* 3 */ };
                /* 4 */ local o2 = {
                /* 5 */     f1: 2,
                /* 6 */     f2: super.f1
                /* 7 */ };
                /* 8 */ o1 + o2
                """
            )
        )

        # NOTE: In this case, although `super.f1` on line 6 references `f1: 1` on line
        # 2, our current 1-pass indexing approach cannot detect this reference. This is
        # because `o2` is defined and indexed before `o1 + o2`, while the `super` scope
        # won't be set up until we index `o1 + o2`.

        self.checkDefRefs(
            FakeWorkspace.single_doc(t),
            def_location=t.location_of("o1"),
            ref_locations=[t.location_of("o1", line=8)],
        )

        self.checkDefRefs(
            FakeWorkspace.single_doc(t),
            def_location=t.location_of("o2", line=4),
            ref_locations=[t.location_of("o2", line=8)],
        )

    def test_fn_params(self):
        t = FakeDocument(
            dedent(
                """\
                local f(p) =
                    p + 1;
                f(1)
                """
            )
        )

        self.checkDefRefs(
            FakeWorkspace.single_doc(t),
            def_location=t.location_of("p"),
            ref_locations=[t.location_of("p", line=2)],
        )

        self.checkDefRefs(
            FakeWorkspace.single_doc(t),
            def_location=t.location_of("f"),
            ref_locations=[t.location_of("f", line=3)],
        )

    def test_fn_param_refs(self):
        t = FakeDocument(
            dedent(
                """\
                function(
                    p1 = p2,
                    p2 = 3,
                    p3 = p1,
                ) p1 + p2 + p3;
                """
            )
        )

        self.checkDefRefs(
            FakeWorkspace.single_doc(t),
            def_location=t.location_of("p1", line=2),
            ref_locations=[
                t.location_of("p1", line=4),
                t.location_of("p1", line=5),
            ],
        )

        self.checkDefRefs(
            FakeWorkspace.single_doc(t),
            def_location=t.location_of("p2", line=3),
            ref_locations=[
                t.location_of("p2", line=2),
                t.location_of("p2", line=5),
            ],
        )

        self.checkDefRefs(
            FakeWorkspace.single_doc(t),
            def_location=t.location_of("p3", line=4),
            ref_locations=[
                t.location_of("p3", line=5),
            ],
        )

    def test_list_comp(self):
        t = FakeDocument("[local y = 1; x + y for x in std.range(1, 3)]")

        self.checkDefRefs(
            FakeWorkspace.single_doc(t),
            def_location=t.location_of("x", nth=2),
            ref_locations=[t.location_of("x")],
        )

        self.checkDefRefs(
            FakeWorkspace.single_doc(t),
            def_location=t.location_of("y"),
            ref_locations=[t.location_of("y", nth=2)],
        )
