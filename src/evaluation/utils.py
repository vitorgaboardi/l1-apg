"""Auxiliary functions to compute main metrics"""
import stanza
from nltk.tree import Tree
from zss import Node, simple_distance

nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', tokenize_no_ssplit=True, use_gpu=True)

def get_syntax_pattern(utterance: str, max_level: int | None = None):
    """Returns a string representing the syntax pattern up to `max_level` levels."""
    doc = nlp_stanza(utterance)
    full_syntax_tree = doc.sentences[0].constituency
    tree = Tree.fromstring(str(full_syntax_tree))

    # Drop artificial ROOT if present
    if tree.label() == "ROOT" and isinstance(tree[0], Tree):
        tree = tree[0]

    def traverse(node, level):
        if max_level is not None and level > max_level:
            return None
        if not isinstance(node, Tree):
            return None  # skip words

        children = [traverse(child, level + 1) for child in node]
        children = [c for c in children if c is not None]

        if children:
            return f"({node.label()} {' '.join(children)})"
        else:
            return f"({node.label()})"

    return traverse(tree, 1)


def compute_tree_edit_distance(pred_parse: str, ref_parse: str) -> int:
    def build_tree(s: str):
        old_t = Tree.fromstring(s)
        root = Node("S")

        def create_tree(curr, t):
            if t.label() and t.label() != "S":
                child = Node(t.label())
                curr.addkid(child)
            else:
                child = curr
            for i in t:
                if isinstance(i, Tree):
                    create_tree(child, i)

        create_tree(root, old_t)
        return root

    return simple_distance(
        build_tree(pred_parse),
        build_tree(ref_parse),
        label_dist=lambda a, b: 0 if a == b else 1
    )
