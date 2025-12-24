import tree_sitter as T
from tree_sitter_language_pack import get_language, get_parser


LANG_JSONNET = get_language("jsonnet")
JSONNET_TS_PARSER = get_parser("jsonnet")


def parse_jsonnet(source: str) -> T.Node:
    return JSONNET_TS_PARSER.parse(source.encode()).root_node
