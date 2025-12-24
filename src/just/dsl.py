from itertools import accumulate, chain, groupby
from pathlib import Path

import lsprotocol.types as L
import tree_sitter as T
from rich.text import Text

from just.ast import (
    AST,
    Arg,
    Bool,
    Document,
    Expr,
    Id,
    Num,
    Param,
    Str,
    merge_locations,
)
from just.parsing import parse_jsonnet
from just.server import WorkspaceIndex


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


class TestDocument:
    def __init__(self, source: str, uri: str = "file:///tmp/test.jsonnet") -> None:
        self.uri = uri
        self.source = source
        self.root = parse_jsonnet(source)
        self.body = Document.from_cst(self.uri, self.root).body

        self.lines = source.splitlines(keepends=True)

        # Appends the last empty line if needed.
        if source.endswith("\n"):
            self.lines.append("")

        # Computes the character offset of the first character in each line, used for
        # converting line-character positions to offsets.
        self.line_offsets = list(accumulate(chain([0], map(len, self.lines))))

    def query_one(self, query: T.Query, capture: str) -> AST:
        [node] = T.QueryCursor(query).captures(self.root).get(capture, [])
        return AST.from_cst(self.uri, node)

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
            lhs.append(ref_doc.highlight([ref_location.range], "black on blue"))

            rhs = [Text.styled("Definition(s)", "black on yellow")]
            for uri, group in groupby(ref_locations, key=lambda loc: loc.uri):
                def_doc = self.docs[uri]
                def_ranges = [loc.range for loc in group]
                rhs.append(def_doc.highlight(def_ranges, "black on blue"))

            rendered.append(side_by_side(Text("\n").join(lhs), Text("\n").join(rhs)))

        return Text("\n\n").join(rendered)
