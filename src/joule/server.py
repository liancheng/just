import dataclasses as D
import logging
from collections import defaultdict
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from typing import Iterator, Sequence

import lsprotocol.types as L
from pygls.lsp.server import LanguageServer

from joule.ast import (
    AST,
    Binary,
    Bind,
    Binding,
    Document,
    Expr,
    Field,
    FieldAccess,
    FixedKey,
    Fn,
    ForSpec,
    Id,
    Import,
    Local,
    Object,
    Operator,
    Scope,
    Self,
    Str,
    Super,
    Visibility,
)
from joule.icon import Icon
from joule.parsing import parse_jsonnet
from joule.typing import URI
from joule.util import head_or_none, maybe
from joule.visitor import Visitor

log = logging.root


@D.dataclass(frozen=True)
class PositionKey:
    pos: L.Position

    def __hash__(self) -> int:
        return hash((self.pos.line, self.pos.character))


@D.dataclass(frozen=True)
class LocationKey:
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

    def __lt__(self, that: "LocationKey") -> bool:
        self_in_other = (
            that.location.range.start <= self.location.range.start
            and self.location.range.end <= that.location.range.end
        )

        return (
            self.location.uri < that.location.uri
            or self.location.uri == that.location.uri
            and self.location.range != that.location.range
            and (
                self_in_other
                or self.location.range.start <= that.location.range.start
                and self.location.range.end <= that.location.range.end
            )
        )


class DocumentIndex(Visitor):
    def __init__(self, workspace_index: "WorkspaceIndex", doc: Document) -> None:
        self.workspace_index = workspace_index
        self.doc: Document = doc
        self.inlay_hints: dict[PositionKey, L.InlayHint] = {}
        self.current_var_scope: Scope = Scope()
        self.current_self_scope: Scope | None = None
        self.current_super_scope: Scope | None = None
        self.root_symbol = L.DocumentSymbol(
            name="__root__",
            kind=L.SymbolKind.File,
            range=doc.location.range,
            selection_range=doc.location.range,
        )
        self.breadcrumbs = [self.root_symbol]
        self.ref_to_defs: LocationMap = defaultdict(list[L.Location])
        self.def_to_refs: LocationMap = defaultdict(list[L.Location])
        self.links: list[L.DocumentLink] = []
        self.hovers: dict[LocationKey, L.Hover] = {}

        self.visit(doc)

    @staticmethod
    def load(
        workspace: WorkspaceIndex,
        uri_or_path: URI | Path,
        source: str | None = None,
    ) -> DocumentIndex:
        match uri_or_path:
            case Path():
                path = uri_or_path
                uri = path.as_uri()
            case _:
                uri = uri_or_path
                path = Path.from_uri(uri)

        source = source or path.read_text("utf-8")
        cst = parse_jsonnet(source)
        doc = Document.from_cst(uri, cst)
        return DocumentIndex(workspace, doc)

    @property
    def uri(self):
        return self.doc.location.uri

    @property
    def document_symbols(self) -> Sequence[L.DocumentSymbol]:
        return self.root_symbol.children or []

    def add_document_symbol(self, symbol: L.DocumentSymbol):
        match (parent := self.breadcrumbs[-1]).children:
            case list():
                parent.children.append(symbol)
            case None:
                parent.children = [symbol]

    def definition(
        self,
        position: L.Position,
        include_current: bool = False,
        local: bool = False,
    ) -> list[L.Location]:
        return self.find_goto_locations(
            position, self.ref_to_defs, include_current, local
        )

    def references(
        self,
        position: L.Position,
        include_current: bool = False,
        local: bool = False,
    ) -> list[L.Location]:
        return self.find_goto_locations(
            position, self.def_to_refs, include_current, local
        )

    def find_goto_locations(
        self,
        position: L.Position,
        lookup: dict[LocationKey, list[L.Location]],
        include_current: bool = False,
        local: bool = False,
    ) -> list[L.Location]:
        candidate = head_or_none(
            (key, locations)
            for key, locations in sorted(lookup.items(), key=lambda pair: pair[0])
            if key.location.range.start <= position <= key.location.range.end
        )

        match candidate:
            case None:
                return []
            case key, locations:
                if local:
                    locations = list(filter(lambda x: x.uri == self.uri, locations))
                if include_current:
                    locations = list(chain([key.location], locations))
                return locations

    def add_reference(self, ref: Expr, binding: Binding):
        defs = self.ref_to_defs[LocationKey(ref.location)]
        refs = self.def_to_refs[LocationKey(binding.location)]

        defs.append(binding.location)
        refs.append(ref.location)

        ref_doc = (
            self
            if ref.location.uri == self.uri
            else self.workspace_index.get_or_load(ref.location.uri)
        )

        def_doc = (
            self
            if binding.location.uri == self.uri
            else self.workspace_index.get_or_load(binding.location.uri)
        )

        ref_doc.add_hint(
            L.InlayHint(
                position=ref.location.range.end,
                label=Icon.Reference,
            )
        )

        # Shows the # of references.
        def_doc.add_hint(
            L.InlayHint(
                position=binding.location.range.end,
                label=f"{Icon.Definition}{len(refs)}",
            )
        )

    def add_hint(self, hint: L.InlayHint):
        self.inlay_hints[PositionKey(hint.position)] = hint

    @contextmanager
    def parent_symbol(self, symbol: L.DocumentSymbol):
        self.breadcrumbs.append(symbol)
        try:
            yield symbol
        finally:
            self.breadcrumbs.pop()

    @contextmanager
    def var_scope(self, scope: Scope):
        prev = self.current_var_scope
        self.current_var_scope = scope
        try:
            yield self.current_var_scope
        finally:
            self.current_var_scope = prev

    @contextmanager
    def super_scope(self, scope: Scope | None):
        prev = self.current_super_scope
        self.current_super_scope = scope
        try:
            yield self.current_super_scope
        finally:
            self.current_super_scope = prev

    @contextmanager
    def self_scope(self, scope: Scope | None):
        prev = self.current_self_scope
        self.current_self_scope = scope
        try:
            yield self.current_self_scope
        finally:
            self.current_self_scope = prev

    def find_self_scope(self, t: AST, scope: Scope) -> Scope | None:
        match t:
            case Object() if t.self_scope is not None:
                return t.self_scope
            case Document():
                return self.find_self_scope(t.body, Scope())
            case Id():
                return head_or_none(
                    self.find_self_scope(target, binding.scope)
                    for binding in maybe(scope.get(t))
                    for target in maybe(binding.target)
                )
            case Field():
                return head_or_none(
                    self.find_self_scope(t.value, var_scope)
                    for obj in maybe(t.enclosing_obj)
                    for var_scope in maybe(obj.var_scope)
                )
            case FieldAccess():
                return head_or_none(
                    self.find_self_scope(target, binding.scope)
                    for parent_scope in maybe(self.find_self_scope(t.obj, scope))
                    for binding in maybe(parent_scope.get(t.field))
                    for target in maybe(binding.target)
                )
            case Binary(_, _, _, rhs):
                return self.find_self_scope(rhs, self.current_var_scope)
            case Import(_, "import", path):
                return head_or_none(
                    self.find_self_scope(index.doc, index.current_var_scope)
                    for index in maybe(self.importee_index(path.raw))
                )
            case Local():
                return head_or_none(
                    self.find_self_scope(t.body, local_scope)
                    for local_scope in maybe(t.scope)
                )
            case Binary(_, Operator.Plus, _, rhs):
                return self.find_self_scope(rhs, scope)
            case Self():
                return t.scope
            case Super():
                return t.scope
            case _:
                return None

    def visit_local(self, e: Local):
        for b in e.binds:
            self.visit_bind(b)

        with self.var_scope(self.current_var_scope.nest()) as nested:
            e.scope = nested
            self.visit(e.body)

    def visit_bind(self, b: Bind):
        self.current_var_scope.bind(b.id.name, b.id.location, b.value)

        match b.value:
            case Fn():
                kind = L.SymbolKind.Function
            case _:
                kind = L.SymbolKind.Variable

        doc_symbol = L.DocumentSymbol(
            name=b.id.name,
            kind=kind,
            range=b.id.location.range,
            selection_range=b.location.range,
        )

        self.add_document_symbol(doc_symbol)
        with self.parent_symbol(doc_symbol):
            with self.var_scope(self.current_var_scope.nest()):
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
        with self.var_scope(self.current_var_scope.nest()):
            for p in e.params:
                self.current_var_scope.bind(p.id.name, p.id.location, p.default)

            for p in e.params:
                doc_symbol = L.DocumentSymbol(
                    name=p.id.name,
                    kind=L.SymbolKind.Variable,
                    range=p.id.location.range,
                    selection_range=p.location.range,
                )

                self.add_document_symbol(doc_symbol)

                if p.default is not None:
                    with self.parent_symbol(doc_symbol):
                        self.visit(p.default)

            self.visit(e.body)

    def visit_for_spec(self, s: ForSpec):
        self.add_document_symbol(
            L.DocumentSymbol(
                name=s.id.name,
                kind=L.SymbolKind.Variable,
                range=s.id.location.range,
                selection_range=s.location.range,
            )
        )

        self.current_var_scope.bind(s.id.name, s.id.location, s.container)

        self.visit(s.container)

    def visit_id(self, e: Id):
        if e.is_variable and (binding := self.current_var_scope.get(e)) is not None:
            self.add_reference(e, binding)

    def visit_field_key(self, e: Object, f: Field):
        assert e.self_scope is not None

        f.enclosing_obj = e

        match f.key:
            case FixedKey(_, Id(_, name)):
                e.self_scope.bind(name, f.key.location, f)
            case FixedKey(_, Str(_, raw)):
                e.self_scope.bind(raw, f.key.location, f)

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
                    padding_left=True,
                    padding_right=True,
                )
            )

    def visit_field_value(self, e: Object, f: Field):
        del e
        self.visit(f.value)

    def resolve_importee_path(self, raw_path: str) -> Path:
        root = Path.from_uri(self.workspace_index.root_uri)

        if raw_path.startswith("../") or raw_path.startswith("./"):
            return Path.from_uri(self.uri).parent.joinpath(raw_path).absolute()
        elif (path := root.joinpath(raw_path).absolute()).exists():
            return path
        else:
            return root.joinpath("vendor", raw_path)

    def importee_index(self, raw_path: str) -> "DocumentIndex | None":
        path = self.resolve_importee_path(raw_path)
        index = self.workspace_index.docs.get(path.as_uri())

        return index or (
            self.workspace_index.get_or_load(
                uri=path.as_uri(),
                source=path.read_text(encoding="utf-8"),
            )
            if path.exists()
            else None
        )

    def visit_import(self, e: Import):
        importee_path = self.resolve_importee_path(e.path.raw).as_uri()

        self.links.append(
            L.DocumentLink(
                range=e.path.location.range,
                target=importee_path,
            )
        )

        self.hovers[LocationKey(e.path.location)] = L.Hover(importee_path)

        self.add_document_symbol(
            L.DocumentSymbol(
                name=e.path.raw,
                kind=L.SymbolKind.File,
                range=e.path.location.range,
                selection_range=e.location.range,
            )
        )

        self.add_hint(
            L.InlayHint(
                position=e.path.location.range.end,
                label=Icon.File,
            )
        )

    def visit_object(self, e: Object):
        e.var_scope = self.current_var_scope
        e.super_scope = self.current_super_scope
        e.self_scope = Scope.empty() if e.super_scope is None else e.super_scope.nest()

        with self.self_scope(e.self_scope):
            super().visit_object(e)

    def visit_field_access(self, e: FieldAccess):
        self.visit(e.obj)

        for scope in maybe(self.find_self_scope(e.obj, self.current_var_scope)):
            for binding in maybe(scope.get(e.field)):
                self.add_reference(e.field, binding)

    def visit_self(self, e: Self):
        e.scope = self.current_self_scope

    def visit_super(self, e: Super):
        e.scope = self.current_super_scope

    def visit_binary(self, e: Binary):
        self.visit(e.lhs)
        base_self = self.find_self_scope(e.lhs, self.current_var_scope)
        with self.super_scope(base_self):
            self.visit(e.rhs)


LocationMap = dict[LocationKey, list[L.Location]]


class WorkspaceIndex:
    def __init__(self, root_uri: URI):
        self.root_uri = root_uri
        self.docs: dict[URI, DocumentIndex] = {}

    def definitions(self, uri: URI, position: L.Position) -> list[L.Location]:
        local_defs = [
            location
            for doc in maybe(self.docs.get(uri))
            for location in doc.definition(position)
        ]

        return local_defs

    def references(self, uri: URI, position: L.Position) -> list[L.Location]:
        return [
            location
            for doc in maybe(self.docs.get(uri))
            for location in doc.references(position)
        ]

    def load(self, uri: URI, source: str | None):
        self.docs[uri] = DocumentIndex.load(self, uri, source)

    def get_or_load(self, uri: URI, source: str | None = None) -> DocumentIndex:
        if uri not in self.docs:
            self.load(uri, source)
        return self.docs[uri]


class JouleLanguageServer(LanguageServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_workspace_root(self, root_path: str, root_uri: URI):
        self.workspace.add_folder(L.WorkspaceFolder(root_path, root_uri))
        self.workspace_index = WorkspaceIndex(root_uri)


server = JouleLanguageServer("just", "v0.1")


@server.feature(L.INITIALIZE)
def initialize(ls: JouleLanguageServer, params: L.InitializeParams):
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
def did_open(ls: JouleLanguageServer, params: L.DidOpenTextDocumentParams):
    doc = params.text_document

    # If the workspace root is not yet discovered, set the root as the parent folder of
    # the first document opened.
    if len(ls.workspace.folders) == 0:
        root = Path.from_uri(doc.uri).absolute().parent
        ls.set_workspace_root(root.as_posix(), root.as_uri())

    ls.workspace_index.load(doc.uri, doc.text)


@server.feature(L.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: JouleLanguageServer, params: L.DidChangeTextDocumentParams):
    # TODO: Patch the AST incrementally.
    doc = ls.workspace.get_text_document(params.text_document.uri)
    ls.workspace_index.load(doc.uri, doc.source)


@server.feature(L.WORKSPACE_SYMBOL)
def workspace_symbol(ls: JouleLanguageServer, _: L.WorkspaceSymbolParams):
    def offsprings(symbol: L.DocumentSymbol) -> Iterator[L.DocumentSymbol]:
        return iter(
            offspring
            for children in maybe(symbol.children)
            for child in children
            for offspring in chain([child], offsprings(child))
        )

    return [
        L.WorkspaceSymbol(
            location=L.Location(doc.uri, symbol.range),
            name=symbol.name,
            kind=symbol.kind,
        )
        for doc in ls.workspace_index.docs.values()
        for top_level_symbol in doc.document_symbols
        for symbol in offsprings(top_level_symbol)
    ]


@server.feature(L.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
def document_symbol(ls: JouleLanguageServer, params: L.DocumentColorParams):
    doc = ls.workspace.get_text_document(params.text_document.uri)
    return ls.workspace_index.get_or_load(doc.uri, doc.source).document_symbols


@server.feature(L.TEXT_DOCUMENT_DEFINITION)
def definition(ls: JouleLanguageServer, params: L.DefinitionParams):
    return ls.workspace_index.definitions(params.text_document.uri, params.position)


@server.feature(L.TEXT_DOCUMENT_REFERENCES)
def references(ls: JouleLanguageServer, params: L.ReferenceParams):
    return ls.workspace_index.references(params.text_document.uri, params.position)


@server.feature(L.TEXT_DOCUMENT_DOCUMENT_HIGHLIGHT)
def document_highlight(ls: JouleLanguageServer, params: L.DocumentHighlightParams):
    doc = ls.workspace.get_text_document(params.text_document.uri)
    doc_index = ls.workspace_index.get_or_load(doc.uri, doc.source)

    defs = doc_index.definition(params.position, include_current=True, local=True)
    refs = doc_index.references(params.position, include_current=True, local=True)

    match defs, refs:
        case [], _:
            return refs
        case _:
            return [r for d in defs for r in doc_index.references(d.range.start)]


@server.feature(L.TEXT_DOCUMENT_INLAY_HINT)
def inlay_hint(ls: JouleLanguageServer, params: L.InlayHintParams):
    doc = ls.workspace.get_text_document(params.text_document.uri)
    doc_index = ls.workspace_index.get_or_load(doc.uri, doc.source)
    return list(doc_index.inlay_hints.values())


@server.feature(L.TEXT_DOCUMENT_DOCUMENT_LINK)
def document_link(ls: JouleLanguageServer, params: L.DocumentLinkParams):
    doc = ls.workspace.get_text_document(params.text_document.uri)
    doc_index = ls.workspace_index.get_or_load(doc.uri, doc.source)
    return doc_index.links


@server.feature(L.TEXT_DOCUMENT_HOVER)
def hover(ls: JouleLanguageServer, params: L.HoverParams):
    doc = ls.workspace.get_text_document(params.text_document.uri)
    doc_index = ls.workspace_index.get_or_load(doc.uri, doc.source)
    return head_or_none(
        hover
        for key, hover in doc_index.hovers.items()
        if key.location.range.start <= params.position <= key.location.range.end
    )
