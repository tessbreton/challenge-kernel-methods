from collections import defaultdict
import itertools
import numpy as np
from tqdm import tqdm

class TrieNodeMismatch:
    def __init__(self, depth=0):
        self.depth = depth
        self.counts = {}
        self.children = defaultdict(self.create_child)

    def create_child(self):
        return TrieNodeMismatch(depth=self.depth + 1)

    def is_leaf(self):
        return len(self.children) == 0

    def __iter__(self):
        yield self

        for child in self.children.values():
            for grandchild in child:
                yield grandchild

class TrieMismatch:
    def __init__(self, seq_size):
        self.root = TrieNodeMismatch()
        self.seq_size = seq_size

    def add(self, id, s, n_miss):
        node = self.root
        for c in s:
            node = node.children[c]

        if id not in node.counts:
            node.counts[id] = n_miss
        else:
            node.counts[id] = min(node.counts[id], n_miss)

    @property
    def nodes(self):
        for node in self.root:
            yield node

    @property
    def num_nodes(self):
        return sum(1 for _ in self.nodes)

    @property
    def leaves(self):
        for node in self.nodes:
            if node.is_leaf():
                yield node

    @property
    def num_leaves(self):
        return sum(1 for _ in self.leaves)

class MismatchKernel:
    def __init__(self, k):
        self.k = k
        self.n_miss = 1
        self.weight = 0.8
        self.trie = TrieMismatch(k)

        self.next_id = 0
        self.fitted_sequences = {}

        self.fitted_ = False
        self.fitted_on_ = None
        self.Letters = ['A', 'C', 'G', 'T']

    def fit(self, S):
        for s in tqdm(S):
            self._fit_string(s)

        self.fitted_ = True
        self.fitted_on_ = S

        return self._build_kernel(S, self.fitted_on_)

    def predict(self, T):
        for t in T:
            self._fit_string(t)

        return self._build_kernel(T, self.fitted_on_)

    def _fit_string(self, s):
        if s in self.fitted_sequences:
            return
        id = self.next_id
        self.next_id += 1
        self.fitted_sequences[s] = id

        for full_sub in self._substrings(s, self.k):
            for i in range(len(full_sub)):
                for letter in self.Letters:
                    missmatch = letter == full_sub[i]
                    full_sub_copy = full_sub[:i] + letter + full_sub[i + 1:]
                    self.trie.add(id, full_sub_copy, int(missmatch))

    def _build_kernel(self, T, S):
        T_ids = [
            self.fitted_sequences[t]
            for t in T
        ]

        S_ids = [
            self.fitted_sequences[s]
            for s in S
        ]

        dot_products = defaultdict(float)
        squared_norms = defaultdict(float)

        weight = self.weight
        set_T_ids = set(T_ids)
        set_S_ids = set(S_ids)
        all_ids = set_T_ids | set_S_ids

        it = dict(iterable=self.trie.leaves, total=self.trie.num_leaves)

        for node in tqdm(**it):
            node_ids = set(node.counts.keys())

            for idx in node_ids & all_ids:
                squared_norms[idx] += weight ** (2 * node.counts[idx])

            for t_idx in node_ids & set_T_ids:
                for s_idx in node_ids & set_S_ids:
                    dot_product = weight ** (node.counts[t_idx] + node.counts[s_idx])
                    dot_products[t_idx, s_idx] += dot_product

        K = np.zeros((len(T), len(S)))

        for i, t_idx in enumerate(T_ids):
            for j, s_idx in enumerate(S_ids):
                K[i, j] = dot_products[t_idx, s_idx] / np.sqrt(squared_norms[t_idx] * squared_norms[s_idx])

        return K

    @staticmethod
    def _substrings(s, k):
        return [
            s[i:i + k]
            for i in range(len(s) - k + 1)
        ]
    

class TrieNodeSpectrum:
    def __init__(self, depth=0):
        self.depth = depth
        self.counts = defaultdict(int)
        self.children = defaultdict(lambda: TrieNodeSpectrum(depth=self.depth + 1))

    def is_leaf(self):
        return not self.children

    def __iter__(self):
        yield self
        for child in self.children.values():
            yield from child

class TrieSpectrum:
    def __init__(self):
        self.root = TrieNodeSpectrum()

    def add(self, id, s):
        node = self.root
        for c in s:
            node = node.children[c]
        node.counts[id] = 1

    @property
    def nodes(self):
        return iter(self.root)

    @property
    def num_nodes(self):
        return sum(1 for _ in self.nodes)

    @property
    def leaves(self):
        return filter(lambda node: node.is_leaf(), self.root)

    @property
    def num_leaves(self):
        return sum(1 for _ in self.leaves)

class SpectrumKernel:
    def __init__(self, k):
        self.k = k
        self.trie = TrieSpectrum()
        self.next_id = 0
        self.fitted_sequences = {}
        self.fitted_ = False
        self.fitted_on_ = None

    def fit(self, S):
        for s in S:
            self._fit_string(s)
        self.fitted_, self.fitted_on_ = True, S
        return self._build_kernel(S, self.fitted_on_)

    def predict(self, T):
        for t in T:
            self._fit_string(t)
        return self._build_kernel(T, self.fitted_on_)

    def _fit_string(self, s):
        if s in self.fitted_sequences:
            return

        self.fitted_sequences[s] = self.next_id
        self.next_id += 1

        substrings = self._substrings(s, self.k)

        for sub in substrings:
            self.trie.add(self.fitted_sequences[s], sub)

    def _build_kernel(self, T, S):
        T_ids = [self.fitted_sequences[t] for t in T]
        S_ids = [self.fitted_sequences[s] for s in S]
        set_T_ids, set_S_ids, all_ids = set(T_ids), set(S_ids), set(T_ids) | set(S_ids)

        dot_products, squared_norms = defaultdict(float), defaultdict(float)

        iterator = tqdm(self.trie.leaves, total=self.trie.num_leaves)

        for node in iterator:
            node_ids = set(node.counts)

            for idx in node_ids & all_ids:
                squared_norms[idx] += node.counts[idx] ** 2 ** (self.k - node.depth)

            for t_idx in node_ids & set_T_ids:
                for s_idx in node_ids & set_S_ids:
                    dot_products[t_idx, s_idx] += node.counts[t_idx] * node.counts[s_idx] ** (self.k - node.depth)

        return np.array([
            [dot_products[t_idx, s_idx] / np.sqrt(squared_norms[t_idx] * squared_norms[s_idx]) 
             for s_idx in S_ids] for t_idx in T_ids
        ])

    @staticmethod
    def _substrings(s, k):
        return [s[i:i + k] for i in range(len(s) - k + 1)]
    

class TrieNodeSubstring:
    def __init__(self, depth=0):
        self.depth = depth
        self.counts = {}
        self.children = defaultdict(lambda: TrieNodeSubstring(depth + 1))

    def is_leaf(self):
        return not self.children

    def __iter__(self):
        yield self
        for child in self.children.values():
            yield from child

class TrieSubstring:
    def __init__(self, seq_size):
        self.root = TrieNodeSubstring()
        self.seq_size = seq_size

    def add(self, id, s, jumps):
        node = self.root
        for c in s:
            node = node.children[c]
        node.counts[id] = min(node.counts.get(id, jumps), jumps)

    @property
    def nodes(self):
        yield from self.root

    @property
    def leaves(self):
        yield from filter(lambda node: node.is_leaf(), self.nodes)

class SubstringKernel:
    def __init__(self, k):
        self.k = k
        self.jump = 2
        self.weight = 0.8
        self.trie = TrieSubstring(k)
        self.fitted_sequences = {}
        self.next_id = 0
        self.fitted_ = False
        self.fitted_on_ = None

    def fit(self, S):
        for s in tqdm(S):
            self._fit_string(s)
        self.fitted_ = True
        self.fitted_on_ = S
        return self._build_kernel(S, self.fitted_on_)

    def predict(self, T):
        for t in T:
            self._fit_string(t)
        return self._build_kernel(T, self.fitted_on_)

    def _fit_string(self, s):
        if s in self.fitted_sequences:
            return
        self.fitted_sequences[s] = self.next_id
        self.next_id += 1
        
        for full_sub in self._substrings(s, self.k + self.jump):
            for sub_idx in itertools.combinations(range(self.jump + self.k), self.k):
                if sub_idx[0] == 0:
                    sub, jumps = self._make_sub(full_sub, sub_idx)
                    self.trie.add(self.fitted_sequences[s], sub, len(sub) + jumps)

    def _build_kernel(self, T, S):
        T_ids, S_ids = [self.fitted_sequences[t] for t in T], [self.fitted_sequences[s] for s in S]
        set_T_ids, set_S_ids, all_ids = set(T_ids), set(S_ids), set(T_ids) | set(S_ids)
        dot_products, squared_norms = defaultdict(float), defaultdict(float)
        
        for node in tqdm(self.trie.leaves, total=sum(1 for _ in self.trie.leaves)):
            node_ids = set(node.counts.keys())
            for idx in node_ids & all_ids:
                squared_norms[idx] += self.weight ** (2 * node.counts[idx])
            for t_idx in node_ids & set_T_ids:
                for s_idx in node_ids & set_S_ids:
                    dot_products[t_idx, s_idx] += self.weight ** (node.counts[t_idx] + node.counts[s_idx])
        
        return np.array([
            [dot_products[t, s] / np.sqrt(squared_norms[t] * squared_norms[s]) for s in S_ids] for t in T_ids
        ])
    
    @staticmethod
    def _substrings(s, k):
        return [s[i:i + k] for i in range(len(s) - k + 1)]
    
    @staticmethod
    def _make_sub(full_sub, sub_idx):
        sub, jumps = full_sub[sub_idx[0]], sum(sub_idx[i] != sub_idx[i - 1] + 1 for i in range(1, len(sub_idx)))
        for i in sub_idx[1:]:
            sub += full_sub[i]
        return sub, jumps