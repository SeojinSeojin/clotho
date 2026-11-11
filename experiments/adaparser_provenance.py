"""Passive ancestry tracking for AdaParser @232951a; no parsing decisions changed.

Track final trie-node insertion ancestry, including merges, and the ancestry used
at each arrival. Candidate insertions precede all LLM-triggered updates.
"""
from collections import Counter


def instrument(parser):
    trie = parser.trie
    original = {name: getattr(trie, name) for name in ('insert', 'delete', 'search', 'update')}
    from logparser.AdaParser.utils import message_split
    pending = [None]
    arrivals = []
    candidate_insertions = [0]

    def node_for(template):
        node = trie.root
        for token in message_split(template):
            node = node.children[token]
        return node

    def insert(template, logID=None):
        original['insert'](template, logID)
        node = node_for(template)
        origins = {'candidate'} if pending[0] is None else set(pending[0])
        if pending[0] is None:
            candidate_insertions[0] += 1
        node._clotho_origins = set(getattr(node, '_clotho_origins', set())) | origins

    def delete(template):
        if pending[0] is not None:
            pending[0].update(getattr(node_for(template), '_clotho_origins', set()))
        return original['delete'](template)

    def search(message):
        node, flag = original['search'](message)
        arrivals.append(sorted(getattr(node, '_clotho_origins', set())) if flag else ['llm'])
        return node, flag

    def update(template, stop_node, logID):
        pending[0] = {'llm'}
        try:
            return original['update'](template, stop_node, logID)
        finally:
            pending[0] = None

    trie.insert, trie.delete, trie.search, trie.update = insert, delete, search, update

    def result():
        final = [None] * len(parser.df_log)
        templates = {}
        duplicate_writes = [0]

        def visit(node):
            if node.is_end_of_token and node.logIDs:
                origin = sorted(node._clotho_origins)
                templates[node.tokens] = origin
                for line_id in node.logIDs:
                    if final[line_id - 1] is not None:
                        duplicate_writes[0] += 1
                    # Upstream outputResult also writes in DFS order: last write wins.
                    final[line_id - 1] = origin
            for child in node.children.values():
                visit(child)
        visit(trie.root)
        assert len(arrivals) == len(final)
        missing = sum(x is None for x in final)
        final = [x if x is not None else ['unknown'] for x in final]
        return {'version': 1, 'candidate_insertions': candidate_insertions[0],
                'arrival_origins': arrivals, 'final_origins': final,
                'duplicate_line_writes': duplicate_writes[0], 'unassigned_lines': missing,
                'template_ancestry': templates}
    return result
