import dataclasses as D
import logging
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import lsprotocol.types as L
from pygls.lsp.server import LanguageServer

from just.ast import (
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
    Visitor,
)
from just.parsing import parse_jsonnet
from just.util import first, head_or_none, maybe

log = logging.root


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

    def __lt__(self, that: "HashableLocation") -> bool:
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
    def __init__(self, workspace_index: "WorkspaceIndex", tree: Document) -> None:
        self.workspace_index = workspace_index
        self.tree: Document = tree
        self.inlay_hints: dict[HashablePosition, L.InlayHint] = {}
        self.current_var_scope: Scope = Scope()
        self.current_self_scope: Scope | None = None
        self.current_super_scope: Scope | None = None
        self.root_symbol = L.DocumentSymbol(
            name="__root__",
            kind=L.SymbolKind.File,
            range=tree.location.range,
            selection_range=tree.location.range,
        )
        self.breadcrumbs = [self.root_symbol]

        # For go-to-definition and find-references
        self.ref_to_defs: LocationMap = defaultdict(list[L.Location])
        self.def_to_refs: LocationMap = defaultdict(list[L.Location])

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

    def find_goto_locations(
        self,
        position: L.Position,
        lookup: dict[HashableLocation, list[L.Location]],
    ) -> list[L.Location]:
        return first(
            lookup[key]
            for key in sorted(lookup.keys())
            if key.location.range.start <= position <= key.location.range.end
        ).or_else([])

    def add_reference(self, ref: Expr, binding: Binding):
        defs = self.ref_to_defs[HashableLocation(ref.location)]
        refs = self.def_to_refs[HashableLocation(binding.location)]

        defs.append(binding.location)
        refs.append(ref.location)

        self.add_hint(
            L.InlayHint(
                position=ref.location.range.start,
                label="󰁝",
                kind=L.InlayHintKind.Parameter,
            )
        )

        # Shows the # of references.
        self.add_hint(
            L.InlayHint(
                position=binding.location.range.start,
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
            case Object() as obj if obj.self_scope is not None:
                return obj.self_scope
            case Document():
                return self.find_self_scope(t.body, Scope())
            case Id() as id:
                return head_or_none(
                    self.find_self_scope(value, binding.scope)
                    for binding in maybe(scope.get(id))
                    for value in maybe(binding.value)
                )
            case Field() as f:
                return head_or_none(
                    self.find_self_scope(f.value, var_scope)
                    for obj in maybe(f.enclosing_obj)
                    for var_scope in maybe(obj.var_scope)
                )
            case FieldAccess() as f:
                return head_or_none(
                    self.find_self_scope(value, binding.scope)
                    for parent_scope in maybe(self.find_self_scope(f.obj, scope))
                    for binding in maybe(parent_scope.get(f.field))
                    for value in maybe(binding.value)
                )
            case Binary(_, _, _, rhs):
                return self.find_self_scope(rhs, self.current_var_scope)
            case Import(_, "import", path):
                return head_or_none(
                    self.find_self_scope(index.tree, index.current_var_scope)
                    for index in maybe(self.importee_index(path.raw))
                )
            case Local() as l:
                return head_or_none(
                    self.find_self_scope(l.body, local_scope)
                    for local_scope in maybe(l.scope)
                )
            case Binary(_, Operator.Plus, _, rhs):
                return self.find_self_scope(rhs, scope)
            case Self() as e:
                return e.scope
            case Super() as e:
                return e.scope
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

        _, doc_symbol = self.add_symbol(
            name=b.id.name,
            kind=kind,
            location=b.id.location,
            selection_range=b.location.range,
        )

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
        if e.is_variable and (binding := self.current_var_scope.get(e)) is not None:
            self.add_reference(e, binding)

    def visit_field_key(self, e: Object, f: Field):
        assert e.self_scope is not None

        f.enclosing_obj = e

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

    def visit_field_value(self, e: Object, f: Field):
        del e
        self.visit(f.value)

    def resolve_importee_path(self, raw_path: str) -> Path:
        root = Path.from_uri(self.workspace_index.root_uri)

        if raw_path.startswith("../") or raw_path.startswith("./"):
            return Path.from_uri(self.uri).parent.joinpath(raw_path).absolute()
        else:
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

    def definitions(self, uri: str, position: L.Position) -> list[L.Location]:
        return [
            location
            for doc in maybe(self.docs.get(uri))
            for location in doc.find_goto_locations(position, doc.ref_to_defs)
        ]

    def references(self, uri: str, position: L.Position) -> list[L.Location]:
        return [
            location
            for doc in maybe(self.docs.get(uri))
            for location in doc.find_goto_locations(position, doc.def_to_refs)
        ]

    def sync(self, uri: str, source: str):
        cst = parse_jsonnet(source)
        doc = Document.from_cst(uri, cst)
        self.docs[uri] = DocumentIndex(self, doc)

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
