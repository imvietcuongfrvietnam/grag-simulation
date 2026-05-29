"""Graph visualization utilities using matplotlib and networkx.

All plotting functions are designed to work without a display (they create
figures but do not call ``plt.show()`` automatically, so they are safe in
headless / CI environments).  Pass ``ax`` to embed in an existing figure.

ASCII helpers (``ascii_graph``, ``ascii_tree``) have no matplotlib dependency
and can be used in any environment.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_TYPE_COLOURS: Dict[str, str] = {
    "PERSON": "#4C72B0",
    "ORG": "#DD8452",
    "LOCATION": "#55A868",
    "EVENT": "#C44E52",
    "CONCEPT": "#8172B3",
    "COMMUNITY": "#937860",
    "UNKNOWN": "#AAAAAA",
}

_COMMUNITY_PALETTE: List[str] = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]


def _get_community_colour(community_id: int) -> str:
    return _COMMUNITY_PALETTE[community_id % len(_COMMUNITY_PALETTE)]


# ---------------------------------------------------------------------------
# Core graph plot
# ---------------------------------------------------------------------------


def plot_knowledge_graph(
    graph: Any,
    title: str = "Knowledge Graph",
    color_by: str = "type",
    ax: Optional[Any] = None,
    max_nodes: int = 80,
    node_size: int = 600,
    font_size: int = 7,
    edge_alpha: float = 0.4,
    figsize: Tuple[int, int] = (14, 10),
) -> Optional[Any]:
    """Plot a KnowledgeGraph with nodes coloured by entity type.

    Parameters
    ----------
    graph:
        A :class:`~grag.core.graph.KnowledgeGraph` instance.
    title:
        Figure title.
    color_by:
        ``'type'`` – colour by ``entity.type``; ``'community'`` – colour by
        community ID (requires community assignment).
    ax:
        Optional matplotlib axes.  If None a new figure/axes pair is created.
    max_nodes:
        Cap the number of plotted nodes to avoid clutter.
    node_size:
        Marker area passed to NetworkX's draw functions.
    font_size:
        Label font size.
    edge_alpha:
        Transparency of edge lines.
    figsize:
        Figure size in inches (width, height).

    Returns
    -------
    The matplotlib :class:`~matplotlib.axes.Axes` object, or ``None`` if
    matplotlib is not available.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("matplotlib / networkx not available.  Install them to use plot_knowledge_graph().")
        return None

    nx_graph = graph._nx

    # Cap to max_nodes for readability
    all_nodes = list(nx_graph.nodes())
    if len(all_nodes) > max_nodes:
        # Keep highest-degree nodes
        degrees = dict(nx_graph.degree())
        all_nodes = sorted(all_nodes, key=lambda n: degrees.get(n, 0), reverse=True)[:max_nodes]
        nx_graph = nx_graph.subgraph(all_nodes)

    # Colour assignment
    node_colours: List[str] = []
    for node in nx_graph.nodes():
        if color_by == "community":
            comm_id = graph.get_community(node)
            if comm_id is not None:
                node_colours.append(_get_community_colour(comm_id))
            else:
                node_colours.append("#AAAAAA")
        else:
            entity = graph.get_entity(node)
            etype = entity.get("type", "UNKNOWN")
            node_colours.append(_TYPE_COLOURS.get(etype, "#AAAAAA"))

    # Labels: use entity label if available, else node id
    labels = {
        node: graph.get_entity(node).get("label", node)[:20]
        for node in nx_graph.nodes()
    }

    # Layout
    if nx_graph.number_of_nodes() <= 20:
        pos = nx.spring_layout(nx_graph, seed=42, k=2.0)
    elif nx_graph.number_of_nodes() <= 50:
        pos = nx.kamada_kawai_layout(nx_graph)
    else:
        pos = nx.spring_layout(nx_graph, seed=42, iterations=30)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(
        nx_graph, pos, ax=ax, node_color=node_colours,
        node_size=node_size, alpha=0.9,
    )
    nx.draw_networkx_edges(
        nx_graph, pos, ax=ax, alpha=edge_alpha, arrows=True,
        arrowsize=12, edge_color="#666666",
        connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_labels(nx_graph, pos, labels=labels, ax=ax, font_size=font_size)

    # Legend
    if color_by == "type":
        seen_types = {
            graph.get_entity(n).get("type", "UNKNOWN") for n in nx_graph.nodes()
        }
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=_TYPE_COLOURS.get(t, "#AAA"), label=t)
            for t in sorted(seen_types)
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=8)
    elif color_by == "community":
        communities_present = {
            graph.get_community(n) for n in nx_graph.nodes()
            if graph.get_community(n) is not None
        }
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=_get_community_colour(c), label=f"Community {c}")
            for c in sorted(communities_present)
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    return ax


# ---------------------------------------------------------------------------
# Community plot
# ---------------------------------------------------------------------------


def plot_communities(
    graph: Any,
    communities: Dict[int, List[str]],
    title: str = "Community Detection",
    ax: Optional[Any] = None,
    figsize: Tuple[int, int] = (14, 10),
    max_nodes: int = 80,
) -> Optional[Any]:
    """Plot the graph with nodes coloured by their community membership.

    Parameters
    ----------
    graph:
        A :class:`~grag.core.graph.KnowledgeGraph` instance.
    communities:
        Mapping of community_id -> list of node IDs.
    title:
        Figure title.
    ax:
        Optional existing axes.
    figsize:
        Figure size in inches.
    max_nodes:
        Maximum nodes to plot.

    Returns
    -------
    Matplotlib axes or ``None``.
    """
    # Assign communities to the graph for the coloring pass
    node_to_comm: Dict[str, int] = {}
    for comm_id, members in communities.items():
        for node in members:
            node_to_comm[node] = comm_id

    # Temporarily assign communities so plot_knowledge_graph can use them
    original_map = dict(graph._community_map)
    graph.assign_communities(node_to_comm)

    ax = plot_knowledge_graph(
        graph,
        title=title,
        color_by="community",
        ax=ax,
        max_nodes=max_nodes,
        figsize=figsize,
    )

    # Restore original map
    graph._community_map = original_map
    return ax


# ---------------------------------------------------------------------------
# RAPTOR tree plot
# ---------------------------------------------------------------------------


def plot_raptor_tree(
    tree: Dict[str, Any],
    title: str = "RAPTOR Tree",
    figsize: Tuple[int, int] = (16, 10),
    node_size: int = 800,
    font_size: int = 8,
) -> Optional[Any]:
    """Plot a RAPTOR hierarchical tree structure.

    The *tree* dict is expected to match the output of RAPTOR's internal tree
    representation:

    .. code-block:: python

        tree = {
            "root": {
                "id": "root",
                "summary": "Root summary ...",
                "children": ["layer1_a", "layer1_b"],
            },
            "layer1_a": {
                "id": "layer1_a",
                "summary": "Layer 1 node A ...",
                "children": ["leaf_x", "leaf_y"],
            },
            # ... leaf nodes have no "children" key or an empty list
        }

    Parameters
    ----------
    tree:
        RAPTOR tree dict (node_id -> node_dict with at least ``'id'`` and
        optional ``'children'`` and ``'summary'`` keys).
    title:
        Figure title.
    figsize:
        Figure size in inches.
    node_size:
        Node marker size.
    font_size:
        Label font size.

    Returns
    -------
    Matplotlib axes or ``None``.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("matplotlib is required for plot_raptor_tree().")
        return None

    if not tree:
        return None

    # Build a directed tree graph
    T = nx.DiGraph()
    for node_id, node_data in tree.items():
        label = str(node_data.get("summary", node_id))[:30]
        T.add_node(node_id, label=label)

    for node_id, node_data in tree.items():
        for child in node_data.get("children", []):
            if child in tree:
                T.add_edge(node_id, child)

    # Determine tree root (node with in-degree 0)
    roots = [n for n, d in T.in_degree() if d == 0]
    root = roots[0] if roots else (list(tree.keys())[0] if tree else None)
    if root is None:
        return None

    # Compute hierarchical layout
    pos = _hierarchical_layout(T, root)

    # Colour by depth
    depths: Dict[str, int] = {}
    _compute_depths(T, root, 0, depths)
    max_depth = max(depths.values()) if depths else 1
    cmap = plt.get_cmap("Blues")
    node_colours = [
        cmap(0.3 + 0.7 * depths.get(n, 0) / max(max_depth, 1))
        for n in T.nodes()
    ]

    labels = {n: T.nodes[n].get("label", n) for n in T.nodes()}

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(T, pos, ax=ax, node_color=node_colours,
                           node_size=node_size, alpha=0.9)
    nx.draw_networkx_edges(T, pos, ax=ax, arrows=True, arrowsize=14,
                           edge_color="#555555", alpha=0.6)
    nx.draw_networkx_labels(T, pos, labels=labels, ax=ax, font_size=font_size)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    return ax


# ---------------------------------------------------------------------------
# ASCII helpers
# ---------------------------------------------------------------------------


def ascii_graph(graph: Any, max_nodes: int = 20) -> str:
    """Return an ASCII adjacency-list representation of the graph.

    Parameters
    ----------
    graph:
        A :class:`~grag.core.graph.KnowledgeGraph` instance.
    max_nodes:
        Maximum number of nodes to include.

    Returns
    -------
    Multi-line string.
    """
    nodes = list(graph.nodes)
    if not nodes:
        return "(empty graph)"

    # Prioritise highest-degree nodes
    try:
        degrees = {n: len(graph.get_neighbors(n, direction="both")) for n in nodes}
        nodes = sorted(nodes, key=lambda n: degrees.get(n, 0), reverse=True)[:max_nodes]
    except Exception:
        nodes = nodes[:max_nodes]

    lines: List[str] = [
        f"Knowledge Graph  ({graph.num_nodes} nodes, {graph.num_edges} edges)",
        "-" * 60,
    ]

    for node in nodes:
        entity = graph.get_entity(node)
        label = entity.get("label", node)
        etype = entity.get("type", "?")
        comm = graph.get_community(node)
        comm_str = f" [C{comm}]" if comm is not None else ""
        neighbours_out = graph.get_neighbors(node, direction="out")
        neighbours_in = graph.get_neighbors(node, direction="in")

        lines.append(f"  ({etype}) {label}{comm_str}")
        for nb in neighbours_out[:5]:
            rel = graph.get_relation(node, nb)
            rel_type = rel.get("relation_type", "->")
            nb_label = graph.get_entity(nb).get("label", nb)
            lines.append(f"    --[{rel_type}]--> {nb_label}")
        for nb in neighbours_in[:3]:
            rel = graph.get_relation(nb, node)
            rel_type = rel.get("relation_type", "<-")
            nb_label = graph.get_entity(nb).get("label", nb)
            lines.append(f"    <--[{rel_type}]-- {nb_label}")

    if graph.num_nodes > max_nodes:
        lines.append(f"  ... ({graph.num_nodes - max_nodes} more nodes not shown)")

    return "\n".join(lines)


def ascii_tree(
    tree: Dict[str, Any],
    node: str = "root",
    prefix: str = "",
    is_last: bool = True,
) -> str:
    """Return an ASCII tree representation of a RAPTOR-style nested dict.

    Parameters
    ----------
    tree:
        Dict mapping node_id -> node_dict (with optional ``'children'`` list
        and ``'summary'`` / ``'label'`` string).
    node:
        The ID of the node to start from (root of the subtree).
    prefix:
        Indentation prefix string (used during recursion).
    is_last:
        Whether this node is the last child of its parent (affects connector).

    Returns
    -------
    Multi-line string.

    Example
    -------
    >>> tree = {
    ...     "root": {"summary": "All topics", "children": ["a", "b"]},
    ...     "a": {"summary": "Topic A", "children": ["a1"]},
    ...     "a1": {"summary": "Sub A1", "children": []},
    ...     "b": {"summary": "Topic B", "children": []},
    ... }
    >>> print(ascii_tree(tree))
    root: All topics
    ├── a: Topic A
    │   └── a1: Sub A1
    └── b: Topic B
    """
    if node not in tree:
        return f"(node '{node}' not found in tree)"

    node_data = tree[node]
    summary = str(node_data.get("summary", node_data.get("label", node)))[:60]
    connector = "└── " if is_last else "├── "
    line = (prefix + connector if prefix else "") + f"{node}: {summary}"

    children = node_data.get("children", [])
    # Filter to children that actually exist in tree
    children = [c for c in children if c in tree]

    child_prefix = prefix + ("    " if is_last else "│   ")
    child_lines: List[str] = []
    for i, child in enumerate(children):
        child_lines.append(
            ascii_tree(tree, node=child, prefix=child_prefix, is_last=(i == len(children) - 1))
        )

    return "\n".join([line] + child_lines)


# ---------------------------------------------------------------------------
# Internal layout helpers
# ---------------------------------------------------------------------------


def _hierarchical_layout(
    T: Any,
    root: str,
    width: float = 2.0,
    vert_gap: float = 1.0,
    vert_loc: float = 0.0,
    xcenter: float = 0.5,
) -> Dict[str, Tuple[float, float]]:
    """Compute a top-down hierarchical layout for a tree (no graphviz required)."""
    pos: Dict[str, Tuple[float, float]] = {}

    def _recurse(node: str, left: float, right: float, depth: int) -> None:
        children = list(T.successors(node))
        pos[node] = ((left + right) / 2.0, -depth * vert_gap)
        if children:
            step = (right - left) / len(children)
            for i, child in enumerate(children):
                _recurse(child, left + i * step, left + (i + 1) * step, depth + 1)

    _recurse(root, 0.0, width, 0)
    return pos


def _compute_depths(
    T: Any, node: str, depth: int, depths: Dict[str, int]
) -> None:
    """BFS depth assignment for tree nodes."""
    depths[node] = depth
    for child in T.successors(node):
        _compute_depths(T, child, depth + 1, depths)
