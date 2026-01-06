import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from joule.ast import (
    AST,
    PrettyAST,
    PrettyCST,
)
from joule.parsing import parse_jsonnet
from joule.server import WorkspaceIndex, server

app = typer.Typer(
    no_args_is_help=True,
    rich_markup_mode="markdown",
)

logging.basicConfig(
    filename="/tmp/pygls.log",
    filemode="w",
    level=logging.DEBUG,
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
    ] = False,
):
    if path == Path("-"):
        uri = "/dev/stdin"
        source = sys.stdin.read()
    else:
        uri = path.absolute().as_uri()
        source = path.read_text()

    cst = parse_jsonnet(source)
    ast = AST.from_cst(uri, cst)
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
    WorkspaceIndex(workspace_root.as_uri()).load(path.as_uri(), path.read_text())
