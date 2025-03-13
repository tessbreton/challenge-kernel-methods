from collections import defaultdict
import itertools
import numpy as np
from tqdm import tqdm


# ----------------------- TRIE NODES ---------------------------------------------------------------------------
class TrieNode:
    def __init__(self, depth=0):
        self.depth = depth
        self.counts = defaultdict(int)
        self.children = defaultdict(self.create_child)

    def create_child(self):
        return TrieNode(depth=self.depth + 1)

    def is_leaf(self):
        return not self.children

    def __iter__(self):
        yield self
        for child in self.children.values():
            yield from child

class TrieNodeMismatch(TrieNode):
    def __init__(self, depth=0):
        super().__init__(depth)

    def create_child(self):
        return TrieNodeMismatch(depth=self.depth + 1)

    def __iter__(self):
        yield self
        for child in self.children.values():
            for grandchild in child:
                yield grandchild

class TrieNodeSubstring(TrieNode):
    def __init__(self, depth=0):
        super().__init__(depth)

    def create_child(self):
        return TrieNodeSubstring(depth=self.depth + 1)


# ----------------------- TRIES ---------------------------------------------------------------------------
class Trie:
    def __init__(self):
        self.root = TrieNode()

    def add(self, id, s):
        node = self.root
        for c in s:
            node = node.children[c]
        node.counts[id] = 1

    @property
    def nodes(self):
        yield from self.root

    @property
    def leaves(self):
        yield from filter(lambda node: node.is_leaf(), self.nodes)

class TrieMismatch(Trie):
    def __init__(self, seq_size):
        self.root = TrieNodeMismatch()
        self.seq_size = seq_size

    def add(self, id, s, n_miss):
        node = self.root
        for c in s:
            node = node.children[c]
        node.counts[id] = n_miss if id not in node.counts else min(node.counts[id], n_miss)

class TrieSubstring(Trie):
    def __init__(self, seq_size):
        self.root = TrieNodeSubstring()
        self.seq_size = seq_size

    def add(self, id, s, jumps):
        node = self.root
        for c in s:
            node = node.children[c]
        node.counts[id] = min(node.counts.get(id, jumps), jumps)


# ------------------- KERNELS ---------------------------------------------------------------------------
class Kernel:
    def __init__(self, k):
        self.k = k
        self.fitted_sequences = {}
        self.next_id = 0
        self.fitted_on_ = None
        self.trie = Trie()
        self.weight = 0.8
    
    def _build_kernel(self, T, S):
        T_ids, S_ids = [self.fitted_sequences[t] for t in T], [self.fitted_sequences[s] for s in S]
        set_T_ids, set_S_ids, all_ids = set(T_ids), set(S_ids), set(T_ids) | set(S_ids)
        dot_products, squared_norms = defaultdict(float), defaultdict(float)
        
        for node in tqdm(self.trie.leaves, total=sum(1 for _ in self.trie.leaves)):
            node_ids = set(node.counts)
            for idx in node_ids & all_ids:
                squared_norms[idx] += self.weight ** (2 * node.counts[idx])
            for t_idx in node_ids & set_T_ids:
                for s_idx in node_ids & set_S_ids:
                    dot_products[t_idx, s_idx] += self.weight ** (node.counts[t_idx] + node.counts[s_idx])
        
        return np.array([[dot_products[t, s] / np.sqrt(squared_norms[t] * squared_norms[s]) for s in S_ids] for t in T_ids])
    
    def _fit_string(self, s):
        raise NotImplementedError
    
    def fit(self, S):
        self.fitted_on_ = S
        for s in tqdm(S):
            self._fit_string(s)
        return self._build_kernel(S, self.fitted_on_)
    
    def predict(self, T):
        for t in T:
            self._fit_string(t)
        return self._build_kernel(T, self.fitted_on_)
    
    @staticmethod
    def _substrings(s, k):
        return [s[i:i + k] for i in range(len(s) - k + 1)]

class MismatchKernel(Kernel):
    def __init__(self, k):
        super().__init__(k)
        self.n_miss = 1
        self.trie = TrieMismatch(k)
        self.letters = ['A', 'C', 'G', 'T']

    def _fit_string(self, s):
        if s in self.fitted_sequences:
            return
        id = self.next_id
        self.next_id += 1
        self.fitted_sequences[s] = id

        for full_sub in self._substrings(s, self.k):
            for i in range(len(full_sub)):
                for letter in self.letters:
                    missmatch = (letter == full_sub[i])
                    full_sub_copy = full_sub[:i] + letter + full_sub[i + 1:]
                    self.trie.add(id, full_sub_copy, int(missmatch))

    
class SpectrumKernel(Kernel):
    def _fit_string(self, s):
        if s in self.fitted_sequences:
            return
        self.fitted_sequences[s] = self.next_id
        self.next_id += 1

        for substring in self._substrings(s, self.k):
            self.trie.add(self.fitted_sequences[s], substring)

    def _build_kernel(self, T, S):
        T_ids, S_ids = [self.fitted_sequences[t] for t in T], [self.fitted_sequences[s] for s in S]
        set_T_ids, set_S_ids, all_ids = set(T_ids), set(S_ids), set(T_ids) | set(S_ids)
        dot_products, squared_norms = defaultdict(float), defaultdict(float)

        for node in tqdm(self.trie.leaves, total=sum(1 for _ in self.trie.leaves)):
            node_ids = set(node.counts)
            for idx in node_ids & all_ids:
                squared_norms[idx] += node.counts[idx] ** 2 ** (self.k - node.depth)
            for t_idx in node_ids & set_T_ids:
                for s_idx in node_ids & set_S_ids:
                    dot_products[t_idx, s_idx] += node.counts[t_idx] * node.counts[s_idx] ** (self.k - node.depth)

        return np.array([[dot_products[t, s] / np.sqrt(squared_norms[t] * squared_norms[s]) for s in S_ids] for t in T_ids])


class SubstringKernel(Kernel):
    def __init__(self, k):
        super().__init__(k)
        self.jump = 2
        self.trie = TrieSubstring(k)

    def _fit_string(self, s):
        if s in self.fitted_sequences:
            return
        self.fitted_sequences[s] = self.next_id
        self.next_id += 1
        
        for full_sub in self._substrings(s, self.k + self.jump):
            for sub_idx in itertools.combinations(range(self.k + self.jump), self.k):
                if sub_idx[0] == 0:
                    sub, jumps = self._make_sub(full_sub, sub_idx)
                    self.trie.add(self.fitted_sequences[s], sub, len(sub) + jumps)

    @staticmethod
    def _make_sub(full_sub, sub_idx):
        sub, jumps = full_sub[sub_idx[0]], sum(sub_idx[i] != sub_idx[i - 1] + 1 for i in range(1, len(sub_idx)))
        for i in sub_idx[1:]:
            sub += full_sub[i]
        return sub, jumps