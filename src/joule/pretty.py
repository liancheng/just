class PrettyTree:
    """An abstract class for pretty-printing tree-like structures."""

    def node_text(self) -> str:
        """Returns a single-line string representing a tree node."""
        raise NotImplementedError()

    def children(self) -> list["PrettyTree"]:
        """Returns a list of child nodes."""
        raise NotImplementedError()

    def __repr__(self):
        def grow(lines: list[str], nodes: list[PrettyTree], branches: str = ""):
            for i, node in enumerate(nodes):
                last_child = i == len(nodes) - 1
                new_branch = ".   " if last_child else "|   "
                fork = "`-- " if last_child else "|-- "

                lines.append(f"{branches}{fork}{node.node_text()}")
                grow(lines, node.children(), branches + new_branch)

        lines = [self.node_text()]
        grow(lines, self.children())

        return "\n".join(lines)
