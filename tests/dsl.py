import re
from itertools import accumulate, chain
from pathlib import Path
from textwrap import dedent

import lsprotocol.types as L
import tree_sitter as T
from rich.text import Text

from joule.ast import (
    AST,
    Arg,
    Array,
    Assert,
    Bind,
    Bool,
    Call,
    ComputedKey,
    Document,
    Expr,
    FixedKey,
    Fn,
    ForSpec,
    Id,
    IdKind,
    IfSpec,
    Import,
    Local,
    LocationLike,
    Num,
    Param,
    Str,
)
from joule.parsing import parse_jsonnet
from joule.server import WorkspaceIndex

LOCATION_MARK_PATTERN = re.compile(
    dedent(
        """\
        (?P<indent>[ ]*)
        (?P<range>\\^([.]*\\^)?)
        (?P<mark>:?[a-z0-9-_.]+:?)
        """
    ),
    re.VERBOSE,
)

LOCATION_MARK_LINE_PATTERN = re.compile(
    f"({LOCATION_MARK_PATTERN.pattern})+",
    re.VERBOSE,
)


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


class LocationDSL(L.Location):
    @property
    def start(self):
        return self.range.start

    @property
    def end(self):
        return self.range.end

    def id(self, name: str, kind: IdKind) -> Id:
        return Id(self, name, kind)

    def var(self, name: str) -> Id:
        return self.id(name, IdKind.Var)

    def var_ref(self, name: str) -> Id:
        return self.id(name, IdKind.VarRef)

    def arg_ref(self, name: str) -> Id:
        return self.id(name, IdKind.ArgRef)

    def num(self, value: float | int) -> Num:
        return Num(self, float(value))

    @property
    def true(self) -> Bool:
        return Bool(self, True)

    @property
    def false(self) -> Bool:
        return Bool(self, False)

    def str(self, value: str) -> Str:
        return Str(self, value)

    def fixed_id_key(self, name: str) -> FixedKey:
        return FixedKey(self, self.id(name, IdKind.Field))

    def fixed_str_key(self, name: str) -> FixedKey:
        return FixedKey(self, self.str(name))

    def computed_key(self, expr: Expr) -> ComputedKey:
        return ComputedKey(self, expr)

    def fn(self, params: list[Param], body: Expr) -> Fn:
        return Fn(self, params, body)

    def call(self, fn: Expr, args: list[Arg]) -> Call:
        return Call(self, fn, args)

    def array(self, *values: Expr) -> Array:
        return Array(self, list(values))

    def if_(self, condition: Expr) -> IfSpec:
        return IfSpec(self, condition)

    def for_(self, id: Id, container: Expr) -> ForSpec:
        return ForSpec(self, id, container)

    def assert_(self, condition: Expr, message: Expr | None = None) -> Assert:
        return Assert(self, condition, message)

    def local(self, binds: list[Bind], body: Expr) -> Local:
        return Local(self, binds, body)

    def import_(self, path: Str) -> Import:
        return Import(self, "import", path)

    def importstr(self, path: Str) -> Import:
        return Import(self, "importstr", path)


class FakeDocument:
    def __init__(self, source: str, uri: str = "file:///tmp/test.jsonnet") -> None:
        self.uri = uri
        self.source, self.locations = self.preprocess(source)
        self.root = parse_jsonnet(self.source)
        self.body = Document.from_cst(self.uri, self.root).body
        self.lines = self.source.splitlines(keepends=True)

        # Appends the last empty line if needed.
        if source.endswith("\n"):
            self.lines.append("")

        # Computes the character offset of the first character in each line, used for
        # converting line-character positions to offsets.
        self.line_offsets = list(accumulate(chain([0], map(len, self.lines))))

    def preprocess(self, source: str) -> tuple[str, dict[str, L.Location]]:
        line_no = -1
        source_lines = []
        open_marks: dict[str, L.Position] = {}
        locations: dict[str, L.Location] = {}

        for line in source.splitlines():
            if not re.fullmatch(LOCATION_MARK_LINE_PATTERN, line):
                line_no += 1
                source_lines.append(line)
            else:
                for m in list(re.finditer(LOCATION_MARK_PATTERN, line)):
                    start_char = m.start("range")
                    end_char = m.end("range")
                    raw_mark = m.group("mark")

                    opening = raw_mark.endswith(":")
                    closing = raw_mark.startswith(":")
                    mark = raw_mark.strip(":")

                    if opening:
                        assert not closing, f"Invalid location mark: {raw_mark}"
                        open_marks[mark] = L.Position(line_no, start_char)
                    else:
                        if closing:
                            assert mark in open_marks, f"No matching open mark: {mark}"
                            start = open_marks.pop(mark)
                        else:
                            start = L.Position(line_no, start_char)

                        assert mark not in locations, f"Duplicate mark: {mark}"
                        end = L.Position(line_no, end_char)
                        locations[mark] = L.Location(self.uri, L.Range(start, end))

        assert len(open_marks) == 0, (
            f"Closing mark(s) missing: {', '.join(open_marks.keys())}"
        )

        return "\n".join(source_lines), locations

    def at(self, mark_or_location: str | LocationLike) -> LocationDSL:
        match mark_or_location:
            case str() as mark:
                location = self.locations[mark]
            case AST() as ast:
                location = ast.location
            case L.Location() as location:
                pass

        return LocationDSL(location.uri, location.range)

    def query_one(self, query: T.Query, capture: str) -> AST:
        [node] = T.QueryCursor(query).captures(self.root).get(capture, [])
        return AST.from_cst(self.uri, node)

    def offset_of(self, pos: L.Position) -> int:
        return self.line_offsets[pos.line] + pos.character

    def start_of(self, needle: str, line: int = 1, nth: int = 1) -> L.Position:
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

    def location_of(self, needle: str, line: int = 1, nth: int = 1) -> L.Location:
        start = self.start_of(needle, line, nth)
        end = L.Position(start.line, start.character + len(needle))
        return L.Location(self.uri, L.Range(start, end))

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


class FakeWorkspace:
    def __init__(self, root_uri: str, docs: list[FakeDocument]) -> None:
        self.docs = {doc.uri: doc for doc in docs}
        self.index = WorkspaceIndex(root_uri)

        for doc in docs:
            assert doc.uri not in self.index.docs
            self.index.load(doc.uri, doc.source)

    @staticmethod
    def single_doc(doc: FakeDocument) -> "FakeWorkspace":
        root_uri = Path.from_uri(doc.uri).absolute().parent.as_uri()
        return FakeWorkspace(root_uri, [doc])
