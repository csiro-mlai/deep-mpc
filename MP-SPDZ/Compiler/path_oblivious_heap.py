"""This module contains an implementation of the "Path Oblivious Heap"
oblivious priority queue as proposed by 
`Shi <https://eprint.iacr.org/2019/274.pdf>`_.

Path Oblivious Heap comes in two variants that build on either Path ORAM
or Circuit ORAM. Both variants support inserting an element and extracting
the element with the highest priority in time :math:`O(\max(\log(n) + s, e))` where :math:`n`
is the queue capacity, :math:`s` is the ORAM stash size, and :math:`e` is the ORAM eviction
complexity. Assuming :math:`s = O(1)` and :math:`e = O(\log(n))`, the operations are in :math:`O(\log n)`.
Currently, only the Path ORAM variant is implemented and tested (the :py:class:`PathObliviousHeap`).

Furthermore, the :py:class:`UniquePathObliviousHeap` class implements an :py:func:`~UniquePathObliviousHeap.update`
operation that is comparable to that of :py:class:`HeapQ` from :py:mod:`dijkstra`, in that it ensures
that every value inserted in the queue is unique, and if :py:func:`~UniquePathObliviousHeap.update` is called
with a value that is already in the queue, the priority of that value is updated to be equal
to the new priority.

The following benchmark compares the online time of updating an element in :py:class:`HeapQ` on top of Path
ORAM and updating an element in :py:class:`UniquePathObliviousHeap` on top of Path ORAM. :py:class:`PathObliviousHeap`
indeed seems to outperform HeapQ from around :math:`n = 2^4`.

.. image:: poh-graph.png

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, List, Tuple, Type, TypeVar

from Compiler import library as lib, oram, util
from Compiler.circuit_oram import CircuitORAM
from Compiler.dijkstra import HeapEntry
from Compiler.path_oram import Counter, PathORAM
from Compiler.types import (
    _arithmetic_register,
    _clear,
    _secret,
    Array,
    cint,
    MemValue,
    regint,
    sint,
)

# Possible extensions:
# - Type hiding security
# - Implement circuit variant

### SETTINGS ###

# Crash if extract_min is called on an empty queue
CRASH_ON_EMPTY = True

# If enabled, compile-time debugging info is printed.
COMPILE_DEBUG = True

# If enabled, high-level debugging messages are printed at runtime.
# Warning: Reveals operation types.
DEBUG = False

# If enabled, low-level trace is printed at runtime
# Warning: Reveals secret information.
TRACE = False

# DEBUG is enabled if TRACE is enabled
DEBUG = DEBUG or TRACE

# Print indentation
print_indent_level = 0
INDENT = "  "


def indent(delta=1):
    """Indent debug printing by `delta`."""
    global print_indent_level
    print_indent_level += delta


def outdent(delta=1):
    """Outdent debug printing by `delta`."""
    indent(delta=-delta)


def indent_string(s):
    global print_indent_level
    return INDENT * print_indent_level + str(s)


def dprint(s, *args, **kwargs):
    """Compile-time debug printing."""
    if COMPILE_DEBUG:
        print(indent_string(s), *args, **kwargs)


def dprint_ln(s, *args, **kwargs):
    """Runtime debug printing.
    To avoid revealing arguments, check TRACE or DEBUG outside this
    function.
    """
    lib.print_ln(indent_string(s), *args, **kwargs)


def dprint_ln_if(cond, s, *args, **kwargs):
    """Runtime conditional debug printing.
    To avoid revealing arguments, check TRACE or DEBUG outside this
    function.
    """
    lib.print_ln_if(cond, indent_string(s), *args, **kwargs)


def dprint_str(s, *args, **kwargs):
    """Runtime debug printing without line break.
    To avoid revealing arguments, check TRACE or DEBUG outside this
    function.
    """
    lib.print_str(indent_string(s), *args, **kwargs)


### IMPLEMENTATION ###

# Types
T = TypeVar("T", _arithmetic_register, int)
_Secret = Type[_secret]


# Utility functions
def random_block(length, value_type):
    if length == 0:
        return value_type(0)
    else:
        return oram.random_block(length, value_type)


class AbstractMinPriorityQueue(ABC, Generic[T]):
    """An abstract class defining the basic behavior
    of a min priority queue.
    """

    @abstractmethod
    def insert(self, value: T, priority: T) -> None:
        """Insert a value with a priority into the queue."""
        pass

    @abstractmethod
    def extract_min(self) -> T:
        """Remove the minimal element in the queue and return it."""
        pass


class EmptyIndexStructure:
    """Since Path Oblivious Heap does not need to
    maintain a position map, we use an empty index structure
    for compatibility.
    """

    def __init__(*args, **kwargs):
        pass

    def noop(*args, **kwargs):
        return None

    def __getattr__(self, _):
        return self.noop


class NoIndexORAM:
    index_structure = EmptyIndexStructure


class SubtreeMinEntry(HeapEntry):
    fields = ["empty", "leaf", "prio", "value"]

    empty: _secret | MemValue
    leaf: _secret | MemValue
    prio: _secret | MemValue
    value: _secret | MemValue

    def __init__(
        self,
        value_type: _Secret,
        empty: _secret | int,
        leaf: _secret | int,
        prio: _secret | int,
        value: _secret | int,
        mem: bool = False,
    ):
        empty = value_type.hard_conv(empty)
        leaf = value_type.hard_conv(leaf)
        prio = value_type.hard_conv(prio)
        value = value_type.hard_conv(value)
        if mem:
            empty = MemValue(empty)
            leaf = MemValue(leaf)
            prio = MemValue(prio)
            value = MemValue(value)
        super().__init__(value_type, empty, leaf, prio, value)
        self.value_type = value_type
        self.mem = mem

    def __eq__(self, other: SubtreeMinEntry) -> _secret:
        """Return 1 if both are empty or if
        (both are non-empty and prio and value are equal).
        """
        both_empty = self.empty * other.empty
        return both_empty + (1 - both_empty) * (
            (self.empty == other.empty)
            * (self.prio == other.prio)
            * (self.value == other.value)
        )

    def value_cmp(self, other: SubtreeMinEntry) -> _secret:
        """Return 1 if both are non-empty and have the same value."""
        return (1 - self.empty) * (1 - other.empty) * (self.value == other.value)

    def __lt__(self, other: SubtreeMinEntry) -> _secret:
        """Entries are always equal if they are empty.
        Otherwise, compare on emptiness,
        then on priority, and finally tie break on value.
        Returns 1 if first has strictly higher priority (smaller value),
        and 0 otherwise.
        """
        # TODO: Tie break is probably not completely secure if there are duplicates.
        # Can be fixed with unique ids

        prio_lt = self.prio < other.prio
        prio_eq = self.prio == other.prio
        value_lt = self.value < other.value

        # If self is empty, it is never less than other.
        # Otherwise, it is less than other if other is empty.
        # Otherwise, it is less than other
        return (1 - self.empty) * (
            other.empty + (1 - other.empty) * (prio_lt + prio_eq * value_lt)
        )

    def __gt__(self, other: SubtreeMinEntry) -> _secret:
        return other < self

    def __le__(self, other: SubtreeMinEntry) -> _secret:
        return (self == other).max(self < other)

    def __ge__(self, other: SubtreeMinEntry) -> _secret:
        return (self == other).max(self > other)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        if self.mem:
            self.__dict__[key].write(value)
        else:
            self.__dict__[key] = value

    @staticmethod
    def get_empty(value_type: _Secret, mem: bool = False) -> SubtreeMinEntry:
        return SubtreeMinEntry(
            value_type,
            value_type(1),
            value_type(0),
            value_type(0),
            value_type(0),
            mem=mem,
        )

    @staticmethod
    def from_entry(entry: oram.Entry, mem: bool = False) -> SubtreeMinEntry:
        """Convert a RAM entry containing the fields
        [empty, index, prio, value, leaf] into a SubtreeMinEntry.
        """
        entry = iter(entry)
        empty = next(entry)
        next(entry)  # disregard index
        leaf = next(entry)
        prio = next(entry)
        value = next(entry)
        return SubtreeMinEntry(value.basic_type, empty, leaf, prio, value, mem=mem)

    def to_entry(self) -> oram.Entry:
        return oram.Entry(
            0,  # Index is not used
            (self.leaf, self.prio, self.value),
            empty=self.empty,
            value_type=self.value_type,
        )

    def write_if(self, cond, new) -> None:
        """Conditional overwriting by a new entry."""
        for field in self.fields:
            self[field] = cond * new[field] + (1 - cond) * self[field]

    def dump(self, str="", indent=True):
        """Reveal contents of entry (insecure)."""
        if TRACE:
            if indent:
                dprint_ln(
                    str + "empty %s, leaf %s, prio %s, value %s",
                    *(x.reveal() for x in self),
                )
            else:
                lib.print_ln(
                    str + "empty %s, leaf %s, prio %s, value %s",
                    *(x.reveal() for x in self),
                )


class BasicMinTree(NoIndexORAM):
    """Basic Min tree data structure behavior."""

    def __init__(self, init_rounds=-1):
        # Maintain subtree-mins in a separate RAM
        # (some of the attributes we access are defined in the ORAM classes,
        # so no meta information is available when accessed in this constructor.)
        empty_min_entry = self._get_empty_entry()
        self.subtree_mins = oram.RAM(
            2 ** (self.D + 1) + 1,  # +1 to make space for stash min (index -1)
            empty_min_entry.types(),
            self.get_array,
        )
        if init_rounds != -1:
            lib.stop_timer()
            lib.start_timer(1)
        self.subtree_mins.init_mem(empty_min_entry)
        if init_rounds != -1:
            lib.stop_timer(1)
            lib.start_timer()

        @lib.function_block
        def evict(leaf: self.value_type.clear_type):
            """Eviction reused from PathORAM,
            but this version accepts a leaf as input.
            """

            if DEBUG:
                dprint_ln("[POH] evict: along path with label %s", leaf.reveal())

            self.use_shuffle_evict = True

            self.state.write(self.value_type(leaf))

            # load the path to temp storage
            # and empty all buckets
            for i, ram_indices in enumerate(self.bucket_indices_on_path_to(leaf)):
                for j, ram_index in enumerate(ram_indices):
                    self.temp_storage[i * self.bucket_size + j] = self.buckets[
                        ram_index
                    ]
                    self.temp_levels[i * self.bucket_size + j] = i
                    self.buckets[ram_index] = self._get_empty_entry()

            # load the stash to temp storage
            # and empty the stash
            for i in range(len(self.stash.ram)):
                self.temp_levels[i + self.bucket_size * (self.D + 1)] = 0
            # for i, entry in enumerate(self.stash.ram):
            @lib.for_range(len(self.stash.ram))
            def f(i):
                entry = self.stash.ram[i]
                self.temp_storage[i + self.bucket_size * (self.D + 1)] = entry

                self.stash.ram[i] = self._get_empty_entry()

            self.path_regs = [None] * self.bucket_size * (self.D + 1)
            self.stash_regs = [None] * len(self.stash.ram)

            for i, ram_indices in enumerate(self.bucket_indices_on_path_to(leaf)):
                for j, ram_index in enumerate(ram_indices):
                    self.path_regs[j + i * self.bucket_size] = self.buckets[ram_index]
            for i in range(len(self.stash.ram)):
                self.stash_regs[i] = self.stash.ram[i]

            # self.sizes = [Counter(0, max_val=4) for i in range(self.D + 1)]
            if self.use_shuffle_evict:
                if self.bucket_size == 4:
                    self.size_bits = [
                        [self.value_type.bit_type(i) for i in (0, 0, 0, 1)]
                        for j in range(self.D + 1)
                    ]
                elif self.bucket_size == 2 or self.bucket_size == 3:
                    self.size_bits = [
                        [self.value_type.bit_type(i) for i in (0, 0)]
                        for j in range(self.D + 1)
                    ]
            else:
                self.size_bits = [
                    [self.value_type.bit_type(0) for i in range(self.bucket_size)]
                    for j in range(self.D + 1)
                ]
            self.stash_size = Counter(0, max_val=len(self.stash.ram))

            leaf = self.state.read().reveal()

            if self.use_shuffle_evict:
                # more efficient eviction using permutation networks
                self.shuffle_evict(leaf)
            else:
                # naive eviction method
                for i, (entry, depth) in enumerate(
                    zip(self.temp_storage, self.temp_levels)
                ):
                    self.evict_block(entry, depth, leaf)

                for i, entry in enumerate(self.stash_regs):
                    self.stash.ram[i] = entry
                for i, ram_indices in enumerate(self.bucket_indices_on_path_to(leaf)):
                    for j, ram_index in enumerate(ram_indices):
                        self.buckets[ram_index] = self.path_regs[
                            i * self.bucket_size + j
                        ]

        self.evict_along_path = evict

    @lib.method_block
    def update_min(self, leaf_label: _clear = None) -> None:
        """Update subtree_min entries on the path from the specified leaf
        to the root bucket (and stash) by finding the current min entry
        of every bucket on the path and comparing it to the subtree-mins
        of the bucket's two children.
        """
        if leaf_label is None:
            leaf_label = self.state.read().reveal()
        if DEBUG:
            dprint_ln("[POH] update_min: along path with label %s", leaf_label)
        indices = self._get_reversed_min_indices_and_children_on_path_to(leaf_label)

        # Edge case (leaf): no children to consider if we are at a leaf.
        # However, we must remember to set the leaf label of the entry.
        leaf_ram_index = indices[0][0]
        indent()
        leaf_min = self._get_bucket_min(leaf_ram_index)
        self._set_subtree_min(leaf_min, index=leaf_ram_index)
        outdent()
        if TRACE:
            leaf_min.dump("[POH] update_min: leaf min: ")

        # Iterate through internal path nodes and root
        for c, l, r in indices[1:]:
            if TRACE:
                dprint_ln("[POH] update_min: bucket %s", c)
            indent()
            current = self._get_bucket_min(c)
            left, right = map(self.get_subtree_min, [l, r])
            outdent()
            if TRACE:
                current.dump("[POH] update_min: current: ")
                left.dump("[POH] update_min: left: ")
                right.dump("[POH] update_min: right: ")

            # Take min of the three entries
            new = current
            new.write_if(left < new, left)
            new.write_if(right < new, right)

            if TRACE:
                new.dump("[POH] update_min: updating min to: ")

            # Update subtree_min of current bucket
            self._set_subtree_min(new, index=c)

        # Edge case (stash): the only child of stash is the root
        # so only compare those two.
        if TRACE:
            dprint_ln("[POH] update_min: stash")
        indent()
        stash_min = self._get_stash_min()
        root_min = self.get_subtree_min(0)
        outdent()
        if TRACE:
            stash_min.dump("[POH] update_min: stash min: ")
            root_min.dump("[POH] update_min: root min: ")

        # Take min of root_min and stash_min
        new = stash_min
        new.write_if(root_min < new, root_min)

        if TRACE:
            new.dump("[POH] update_min: updating stash min to: ")

        # Update subtree_min of stash
        self._set_subtree_min(new)

    @lib.method_block
    def insert(
        self, value: _secret, priority: _secret, fake: _secret, empty: _secret = None
    ) -> None:
        """Insert an entry in the stash, assigning it a random leaf label,
        evict along two random, non-overlapping (except in the root) paths,
        and update_min along the two same paths.
        """
        # O(log n)

        if empty is None:
            empty = self.value_type(0)

        # Insert entry into stash with random leaf
        leaf_label = self._get_random_leaf_label()
        if TRACE:
            dprint_ln("[POH] insert: sampled random leaf label %s", leaf_label.reveal())
        self.add(
            oram.Entry(
                MemValue(sint(0)),
                [MemValue(v) for v in (priority, value)],
                empty=fake + (1 - fake) * empty,
                value_type=self.value_type,
            ),
            state=MemValue(leaf_label),
            evict=False,  # We evict along two random, non-overlapping paths later
        )
        if TRACE:
            dprint_ln("[POH] insert: stash:")
            self.dump_stash()

        # Evict and update
        leaf_label_even = None
        leaf_label_odd = None
        indent()
        if self.D == 0:
            # Base case (only stash and root node)
            self.evict_along_path(self.value_type.clear_type(0))
        else:
            # Get two random, non-overlapping leaf paths (except in the root)
            # Due to Path ORAM using the leaf index bits for indexing in reversed
            # order, we need to get a random even and uneven label
            leaf_label_even = random_block(self.D - 1, self.value_type).reveal() * 2
            leaf_label_odd = random_block(self.D - 1, self.value_type).reveal() * 2 + 1
            # Evict along two random non-overlapping paths
            self.evict_along_path(leaf_label_even)
            self.evict_along_path(leaf_label_odd)
        outdent()

        if TRACE:
            dprint_ln("[POH] insert: stash:")
            indent()
            self.dump_stash()
            outdent()
            dprint_ln("[POH] insert: ram:")
            indent()
            self.dump_ram()
            outdent()

        # UpdateMin along same paths
        indent()
        if self.D == 0:
            self.update_min(self.value_type.clear_type(0))
        else:
            self.update_min(leaf_label_even)
            self.update_min(leaf_label_odd)
        outdent()

        return leaf_label

    @lib.method_block
    def update(
        self,
        value: _secret,
        priority: _secret,
        leaf_label: _clear,
        fake: _secret,
        empty=None,
    ):
        """Update an existing value that resides on the path to `leaf`.
        Then evict and update_min along the path.
        Important: Assumes that values in the queue are unique
        (otherwise all entries with the specified value are updated).
        """
        # O(log n)
        if empty is None:
            empty = self.value_type(0)

        new_entry = SubtreeMinEntry(
            self.value_type, empty, leaf_label, priority, value, mem=True
        )

        if TRACE:
            dprint_ln(
                "[POH] update: fake = %s, leaf_label = %s",
                fake.reveal(),
                leaf_label.reveal(),
            )

        # Scan path and remove element (unless fake)
        for i, _, _ in self._get_reversed_min_indices_and_children_on_path_to(
            leaf_label
        ):
            start = i * self.bucket_size
            stop = start + self.bucket_size

            @lib.for_range(start, stop=stop)
            def _(j):
                current_entry = SubtreeMinEntry.from_entry(self.buckets[j], mem=True)
                if TRACE:
                    dprint_str("[POH] update: current element (bucket %s): ", i)
                    current_entry.dump(indent=False)
                found = current_entry.value_cmp(new_entry)
                current_entry.write_if((1 - fake) * found, new_entry)
                self.buckets[j] = current_entry.to_entry()

        # Scan stash and remove element (unless fake)
        @lib.for_range(0, len(self.stash.ram))
        def _(i):
            current_entry = SubtreeMinEntry.from_entry(self.stash.ram[i])
            if TRACE:
                current_entry.dump(f"[POH] update: current element (stash): ")
            found = current_entry.value_cmp(new_entry)
            current_entry.write_if((1 - fake) * found, new_entry)
            self.stash.ram[i] = current_entry.to_entry()

        # evict along path to leaf
        indent()
        self.evict_along_path(leaf_label)
        outdent()

        # update_min along path to leaf
        indent()
        self.update_min(leaf_label)
        outdent()

    @lib.method_block
    def extract_min(self, fake: _secret) -> _secret:
        """Look up subtree-min of stash and extract it by linear scanning the structure.
        Then, evict along the extracted path, and finally, update_min along the path.
        """
        # O(log n)

        # Get min entry from stash
        indent()
        min_entry = self.get_subtree_min()
        outdent()
        if TRACE:
            min_entry.dump("[POH] extract_min: global min entry: ")
        if CRASH_ON_EMPTY:
            empty = min_entry.empty.reveal()
            dprint_ln_if(empty, "[POH] extract_min: empty subtree-min! Crashing...")
            lib.crash(empty)
        random_leaf_label = self._get_random_leaf_label()
        leaf_label = (
            min_entry.empty * random_leaf_label + (1 - min_entry.empty) * min_entry.leaf
        ).reveal()
        empty_entry = SubtreeMinEntry.get_empty(self.value_type)

        if DEBUG:
            dprint_ln("[POH] extract_min: searching path to leaf %s", leaf_label)

        # If duplicates, ensure we only remove one.
        done = MemValue(sint(0))

        # Scan path and remove element (unless fake)
        for i, _, _ in self._get_reversed_min_indices_and_children_on_path_to(
            leaf_label
        ):
            start = i * self.bucket_size
            stop = start + self.bucket_size

            @lib.for_range(start, stop=stop)
            def _(j):
                current_entry = SubtreeMinEntry.from_entry(self.buckets[j])
                if TRACE:
                    dprint_str("[POH] extract_min: current element (bucket %s): ", i)
                    current_entry.dump(indent=False)
                found = min_entry == current_entry
                write = (1 - fake) * found * (1 - done)
                current_entry.write_if(write, empty_entry)
                done.write(found.max(done.read()))
                self.buckets[j] = current_entry.to_entry()

        # Scan stash and remove element (unless fake)
        @lib.for_range(0, len(self.stash.ram))
        def _(i):
            current_entry = SubtreeMinEntry.from_entry(self.stash.ram[i])
            if TRACE:
                current_entry.dump(f"[POH] extract_min: current element (stash): ")
            found = min_entry == current_entry
            current_entry.write_if((1 - fake) * found * (1 - done), empty_entry)
            done.write(found.max(done.read()))
            self.stash.ram[i] = current_entry.to_entry()

        # evict along path to leaf
        indent()
        self.evict_along_path(leaf_label)
        outdent()

        # update_min along path to leaf
        indent()
        self.update_min(leaf_label)
        outdent()
        return min_entry.value

    def _get_empty_entry(self) -> oram.Entry:
        return oram.Entry.get_empty(*self.internal_entry_size())

    def get_subtree_min(self, index: int | _clear = -1) -> SubtreeMinEntry:
        """Returns a SubtreeMinEntry representing the subtree-min
        of the bucket with the specified index. If index is not specified,
        it returns the subtree-min of the stash (index -1),
        which is the subtree-min of the complete tree.
        """
        return SubtreeMinEntry.from_entry(self.subtree_mins[index])

    def _set_subtree_min(
        self, entry: SubtreeMinEntry, index: int | _clear = -1
    ) -> None:
        """Sets the subtree-min of the bucket with the specified index
        to the specified entry. Index defaults to stash (-1).
        """
        self.subtree_mins[index] = entry.to_entry()

    def _get_bucket_min(self, index: _clear) -> SubtreeMinEntry:
        """Get the min entry of a bucket by linear scan."""
        start = index * self.bucket_size
        stop = start + self.bucket_size
        return self._get_ram_min(self.buckets, start, stop)

    def _get_stash_min(self) -> SubtreeMinEntry:
        """Get the min entry of the stash by linear scan."""
        return self._get_ram_min(self.stash.ram, 0, len(self.stash.ram))

    def _get_ram_min(
        self, ram: oram.RAM, start: int | _clear, stop: int | _clear
    ) -> SubtreeMinEntry:
        """Scan through RAM indices, finding the entry with highest priority."""

        # It is very important to set mem=True to use MemValues,
        # so we can access this inside for_range.
        current_min = SubtreeMinEntry.get_empty(self.value_type, mem=True)

        @lib.for_range(start, stop=stop)
        def _(i):
            entry = SubtreeMinEntry.from_entry(ram[i], mem=True)
            entry_min = entry < current_min
            if TRACE:
                current_min.dump("[POH] _get_ram_min: current min: ")
                entry.dump("[POH] _get_ram_min: current entry: ")
                dprint_ln(
                    "[POH] _get_ram_min: current entry < current min: %s",
                    entry_min.reveal(),
                )
            current_min.write_if(entry_min, entry)

        return current_min

    def _get_reversed_min_indices_and_children_on_path_to(
        self, leaf_label: _clear | int
    ) -> List[Tuple[int, int, int]] | List[Tuple[_clear, _clear, _clear]]:
        """Returns a list from leaf to root of tuples of (index, left_child, right_child).
        Used for update_min.
        Note that leaf label bits are used from least to most significant bit,
        so even leaves are indexed first, then odd, e.g. (for 8 leaves):

            leaf_label    leaf_index (left to right)
            000           000 (0)
            001           100 (4)
            010           010 (2)
            011           110 (6)
            100           001 (1)
            101           101 (5)
            110           011 (3)
            111           111 (7)

        In other words, leaf indice bits are reversed.
        """
        leaf_label = regint(leaf_label)
        indices = [(0, 1, 2)]
        index = 0
        for _ in range(self.D):
            index = 2 * index + 1 + (cint(leaf_label) & 1)
            leaf_label >>= 1
            indices += [(index,) + self._get_child_indices(index)]
        # if TRACE:
        #     [dprint_ln("%s", i) for i in indices]
        return list(reversed(indices))

    def _get_child_indices(self, i) -> Tuple[int, int]:
        """This is how a binary tree works."""
        return 2 * i + 1, 2 * i + 2

    def _get_random_leaf_label(self) -> _secret:
        return random_block(self.D, self.value_type)

    def dump_stash(self):
        """Insecure."""
        for i in range(len(self.stash.ram)):
            SubtreeMinEntry.from_entry(self.stash.ram[i]).dump()

    def dump_ram(self):
        """Insecure."""
        for i in range(len(self.ram)):
            if i % self.bucket_size == 0:
                dprint_ln("bucket %s", i // self.bucket_size)
            indent()
            SubtreeMinEntry.from_entry(self.ram[i]).dump()
            outdent()


class CircuitMinTree(CircuitORAM, BasicMinTree):
    """Binary Bucket Tree data structure
    using Circuit ORAM as underlying data structure.

    NOT TESTED.
    """

    def __init__(
        self,
        capacity: int,
        int_type: _Secret = sint,
        entry_size: Tuple[int] | None = None,
        bucket_size: int = 3,
        stash_size: int | None = None,
        init_rounds: int = -1,
    ):
        CircuitORAM.__init__(
            self,
            capacity,
            value_type=int_type,
            entry_size=entry_size,
            bucket_size=bucket_size,
            stash_size=stash_size,
            init_rounds=init_rounds,
        )
        BasicMinTree.__init__(self)


class PathMinTree(PathORAM, BasicMinTree):
    """Binary Bucket Tree data structure
    using Path ORAM as underlying data structure.
    """

    def __init__(
        self,
        capacity: int,
        int_type: _Secret = sint,
        entry_size: Tuple[int] | None = None,
        bucket_oram: oram.AbstractORAM = oram.TrivialORAM,
        bucket_size: int = 2,
        stash_size: int | None = None,
        init_rounds: int = -1,
    ):
        PathORAM.__init__(
            self,
            capacity,
            value_type=int_type,
            entry_size=entry_size,
            bucket_oram=bucket_oram,
            bucket_size=bucket_size,
            stash_size=stash_size,
            init_rounds=init_rounds,
        )
        # For compatibility with inherited __repr__
        self.ram = self.buckets
        self.root = oram.RefBucket(1, self)

        BasicMinTree.__init__(self, init_rounds)


class POHVariant(Enum):
    """Constants representing Path and Circuit variants
    and utility functions to map the variants to defaults.
    """

    PATH = 0
    CIRCUIT = 1

    def get_tree_class(self):
        return PathMinTree if self == self.PATH else CircuitMinTree

    def get_default_bucket_size(self):
        return 2 if self == self.PATH else 3

    def __repr__(self):
        return "Path" if self == self.PATH else "Circuit"


class PathObliviousHeap(AbstractMinPriorityQueue[_secret]):
    """A basic Path Oblivious Heap implementation supporting
    insert, extract_min, and find_min.

    The queue is guaranteed to have at least the specified capacity
    with negligible error probability.

    If inserting more entries than there is capacity for,
    the behavior depends on the value of the flag :py:obj:`oram.crash_on_overflow`.
    If the flag is set, the program crashes. Otherwise, the entry is simply
    not inserted.

    :ivar capacity: The capacity of the queue.
    :ivar type_hiding_security: A boolean indicating whether
        type hiding security is enabled. Enabling this
        makes the cost of every operation equal to the
        sum of the costs of all operations. This is initially
        set by passing an argument to the class constructor.
    :ivar int_type: The secret integer type of entry members.
    :ivar entry_size: A tuple specifying the bit lengths of the entries
        in the order (priority, value).
    :iver tree: The MinTree data structure storing subtree-mins
    """

    def __init__(
        self,
        capacity: int,
        security: int | None = None,
        type_hiding_security: bool = False,
        int_type: _Secret = sint,
        entry_size: Tuple[int] | None = None,
        variant: POHVariant = POHVariant.PATH,
        bucket_oram: oram.AbstractORAM = oram.TrivialORAM,
        bucket_size: int | None = None,
        stash_size: int | None = None,
        init_rounds: int = -1,
    ):
        """
        Initializes a Path Oblivious Heap priority queue.

        :param capacity: The max capacity of the queue.
        :param security: A security parameter, used for determining the stash size
            in order to make the error probability negligible in this parameter.
            Defaults to be equal to the capacity.
        :param type_hiding_security: (Currently not supported) True if the types of
            executed operations should be oblivious, False otherwise. Defaults to False.
        :param int_type: The data type of the queue, used for both key and value.
            Defaults to `sint`.
        :param entry_size: A tuple containing an integer per entry value that specifies
            the bit length of that value. The last tuple index specifies the value size.
            Defaults to `(32, util.log2(capacity))`.
        :param variant: A `POHVariant` enum class member specifying the variant (either
            `PATH` or `CIRCUIT`). Defaults to `PATH`.
        :param bucket_oram: The ORAM used in every bucket. Defaults to `oram.TrivialORAM`.
        :param bucket_size: The size of every bucket. Defaults to
            `variant.get_default_bucket_size()`.
        :param stash_size: The size of the stash. Defaults to the squared base 2 logarithm
            of the security parameter.
        :param init_rounds: If not equal to -1, initialization is timed in isolation.
            Defaults to -1.
        """
        # Check inputs
        if int_type != sint:
            raise lib.CompilerError(
                "[POH] __init__: Only sint is supported as int_type."
            )

        if variant is not POHVariant.PATH:
            raise lib.CompilerError(
                "[POH] __init__: Only the PATH variant is supported."
            )

        # Path ORAM does not support capacity < 2
        capacity = max(capacity, 2)

        # TODO: Figure out what default should be (capacity = poly(security))
        if security is None:
            security = capacity

        # Use default entry size (for Dijkstra) if not specified (distance, node)
        if entry_size is None:
            entry_size = (32, util.log2(capacity))

        # Use default bucket size if not specified
        if bucket_size is None:
            bucket_size = variant.get_default_bucket_size()

        # Theoretically, the stash size should be superlogarithmic
        # in the security parameter. But empirically, a constant size
        # 20 works.
        if stash_size is None:
            stash_size = 20

        # Initialize basic class fields
        self.int_type = int_type
        self.type_hiding_security = type_hiding_security
        self.capacity = capacity
        self.entry_size = entry_size

        # Print debug messages
        dprint(f"[POH] __init__: Initializing a queue...")
        dprint(f"[POH] __init__: Variant is {variant}")
        dprint(f"[POH] __init__: Capacity is {capacity}")
        dprint(f"[POH] __init__: Security is {security}")
        dprint(f"[POH] __init__: Stash size is {stash_size}")
        dprint(f"[POH] __init__: Entry size is {entry_size}")
        dprint(
            f"[POH] __init__: Type hiding security is {'en' if self.type_hiding_security else 'dis'}abled",
        )

        # Initialize data structure with dummy elements
        self.tree = variant.get_tree_class()(
            capacity,
            int_type=int_type,
            entry_size=entry_size,
            bucket_oram=bucket_oram,
            bucket_size=bucket_size,
            stash_size=stash_size,
            init_rounds=init_rounds,
        )

    def insert(self, value, priority, fake: bool = False) -> None:
        """Insert an element with a priority into the queue."""
        value = self.int_type.hard_conv(value)
        priority = self.int_type.hard_conv(priority)
        fake = self.int_type.hard_conv(fake)
        self._insert(value, priority, fake)

    def extract_min(self, fake: bool = False) -> _secret | None:
        """Extract the element with the smallest (ie. highest)
        priority from the queue.
        """
        fake = self.int_type.hard_conv(fake)
        return self._extract_min(fake)

    def find_min(self, fake: bool = False) -> _secret | None:
        """Find the element with the smallest (ie. highest)
        priority in the queue and return its value and priority.
        Returns -1 if empty.
        """
        fake = self.int_type.hard_conv(fake)  # Not supported
        return self._find_min()

    def _insert(self, value: _secret, priority: _secret, fake: _secret) -> None:
        if TRACE:
            dprint_ln(
                "\n[POH] insert: {value: %s, prio: %s}",
                value.reveal(),
                priority.reveal(),
            )
        elif DEBUG:
            dprint_ln("\n[POH] insert")
        indent()
        self.tree.insert(value, priority, fake)
        outdent()

    def _extract_min(self, fake: _secret) -> _secret:
        if DEBUG:
            dprint_ln("\n[POH] extract_min")
        indent()
        value = self.tree.extract_min(fake)
        outdent()
        if TRACE:
            dprint_ln("[POH] extract_min: extracted value %s", value.reveal())
        return value

    def _find_min(self) -> _secret:
        if DEBUG:
            dprint_ln("\n[POH] find_min")
        entry = self.tree.get_subtree_min()
        if TRACE:
            entry.dump("[POH] find_min: found entry: ")
            dprint_ln_if(
                entry.empty.reveal(), "[POH] Found empty entry during find_min!"
            )
        return entry.empty.if_else(self.int_type(-1), entry.value)


class UniquePathObliviousHeap(PathObliviousHeap):
    """A Path Oblivious Heap that ensures that all values in the queue are unique
    and supports updating a value with a new priority by maintaining a value to
    leaf index map using ORAM.
    """

    def __init__(self, *args, oram_type=oram.OptimalORAM, init_rounds=-1, **kwargs):
        super().__init__(*args, init_rounds=init_rounds, **kwargs)
        # Keep track of leaf_labels of every value in the queue
        # Capacity depends on the bit size of values,
        # and entry size needs to be big enough to store a leaf label
        self.value_leaf_index = oram_type(
            2 ** self.entry_size[1],
            entry_size=util.log2(self.capacity),
            init_rounds=init_rounds,
            value_type=self.int_type,
        )

    def update(self, value, priority, empty=0, fake=False):
        """Update the priority of an entry with a given value.
        If such an entry does not already exist, it is inserted.
        """
        value = self.int_type.hard_conv(value)
        priority = self.int_type.hard_conv(priority)
        empty = self.int_type.hard_conv(empty)
        fake = self.int_type.hard_conv(fake)
        self._update(value, priority, empty, fake)

    def _update(
        self, value: _secret, priority: _secret, empty: _secret, fake: _secret
    ) -> None:
        if TRACE:
            dprint_ln(
                "\n[POH] update: {value: %s, prio: %s, empty: %s, fake: %s}",
                value.reveal(),
                priority.reveal(),
                empty.reveal(),
                fake.reveal(),
            )
        elif DEBUG:
            dprint_ln("\n[POH] update")
        leaf_label, not_found = self.value_leaf_index.read(value)
        assert len(leaf_label) == 1
        leaf_label = leaf_label[0]
        if TRACE:
            dprint_ln(
                "[POH] update: leaf_label = %s, not_found = %s",
                leaf_label.reveal(),
                not_found.reveal(),
            )
        random_leaf_label = self.tree._get_random_leaf_label()
        leaf_label = (
            fake.max(not_found) * random_leaf_label
            + (1 - fake.max(not_found)) * leaf_label
        ).reveal()
        indent()
        self.tree.update(value, priority, leaf_label, fake.max(not_found), empty)
        insert_label = self.tree.insert(value, priority, fake.max(1 - not_found), empty)
        self.value_leaf_index.access(
            value,
            not_found * insert_label + (1 - not_found) * leaf_label,
            (1 - fake),
            empty,
        )
        outdent()
        if TRACE:
            dprint_ln("[POH] update: value_leaf_index:")
            dprint_ln("%s", self.value_leaf_index.ram)

    def extract_min(self, fake: bool = False) -> _secret | None:
        value = super().extract_min(fake=fake)
        self.value_leaf_index.access(
            value,
            0,
            (1 - self.int_type.hard_conv(fake)),
            self.int_type(1),
        )
        if TRACE:
            dprint_ln("[POH] extract_min: value_leaf_index:")
            dprint_ln("%s", self.value_leaf_index.ram)
        return value

    def insert(self, value, priority, fake: bool = False) -> None:
        self.update(value, priority, fake=fake)


class POHToHeapQAdapter(PathObliviousHeap):
    """Adapts Path Oblivious Heap to the HeapQ interface,
    allowing plug-and-play replacement in the Dijkstra
    implementation.
    """

    def __init__(
        self,
        max_size,
        *args,
        int_type=sint,
        variant=POHVariant.PATH,
        bucket_size=None,
        stash_size=None,
        init_rounds=-1,
        entry_size=None,
        **kwargs,
    ):
        """Initialize a POH with the required capacity
        and disregard all irrelevant parameters.
        """
        super().__init__(
            max_size,
            int_type=int_type,
            variant=variant,
            bucket_size=bucket_size,
            stash_size=stash_size,
            init_rounds=init_rounds,
            entry_size=entry_size,
        )

    def update(self, value, priority, for_real=True):
        """Call :py:func:`insert` instead of update.
        Warning: When using this adapter, duplicate values are
        allowed to be inserted, and no values are ever updated.
        """
        self.insert(value, priority, fake=(1 - for_real))

    def pop(self, for_real=True):
        """Renaming of pop to :py:func:`extract_min`."""
        return self.extract_min(fake=(1 - for_real))


class UniquePOHToHeapQAdapter(UniquePathObliviousHeap):
    """
    Adapts Unique Path Oblivious Heap to the HeapQ interface,
    allowing plug-and-play replacement in the Dijkstra
    implementation.
    """

    def __init__(
        self,
        max_size,
        *args,
        int_type=sint,
        variant=POHVariant.PATH,
        oram_type=oram.OptimalORAM,
        bucket_size=None,
        stash_size=None,
        init_rounds=-1,
        entry_size=None,
        **kwargs,
    ):
        """Initialize a POH with the required capacity
        and disregard all irrelevant parameters.
        """
        super().__init__(
            max_size,
            int_type=int_type,
            variant=variant,
            oram_type=oram_type,
            bucket_size=bucket_size,
            stash_size=stash_size,
            init_rounds=init_rounds,
            entry_size=entry_size,
        )

    def update(self, value, priority, for_real=True):
        super().update(value, priority, fake=(1 - for_real))

    def pop(self, for_real=True):
        """Renaming of pop to :py:func:`extract_min`."""
        return self.extract_min(fake=(1 - for_real))

    def insert(self, value, priority, for_real=True) -> None:
        self.update(value, priority, for_real=for_real)


def path_oblivious_sort(
    keys: Array,
    values: Array,
    key_length: int,
    value_length: int | None = None,
    **kwargs,
):
    """Sort values in place according to keys using Path Oblivious Heap
    by calling insert followed by extract min.
    """
    assert len(keys) == len(values)
    n = len(keys)
    if value_length is None:
        value_length = key_length
    q = PathObliviousHeap(n, entry_size=(key_length, value_length), **kwargs)

    @lib.for_range(n)
    def _(i):
        q.insert(values[i], keys[i])

    @lib.for_range(n)
    def _(i):
        values[i] = q.extract_min()


def test_SubtreeMinEntry_cmp():
    a = SubtreeMinEntry(sint, 0, 42, 6, 14)
    b = SubtreeMinEntry(sint, 0, 42, 6, 13)
    c = SubtreeMinEntry(sint, 0, 42, 5, 13)
    d = SubtreeMinEntry(sint, 1, 10, 0, 0)
    e = SubtreeMinEntry(sint, 0, 17, 0, 7, mem=True)
    f = SubtreeMinEntry(sint, 0, 17, 0, 6, mem=True)

    dprint("a < a: %s", (a < a).reveal())  # 0
    dprint_ln(", expected output: 0")
    dprint("a > a: %s", (a > a).reveal())  # 0
    dprint_ln(", expected output: 0")
    dprint("a == a: %s", (a == a).reveal())  # 1
    dprint_ln(", expected output: 1")
    dprint("a <= a: %s", (a <= a).reveal())  # 1
    dprint_ln(", expected output: 1")
    dprint("a >= a: %s", (a >= a).reveal())  # 1
    dprint_ln(", expected output: 1")
    dprint("a < b: %s", (a < b).reveal())  # 0
    dprint_ln(", expected output: 0")
    dprint("a == b: %s", (a == b).reveal())  # 0
    dprint_ln(", expected output: 0")
    dprint("b < a: %s", (b < a).reveal())  # 1
    dprint_ln(", expected output: 1")
    dprint("b > a: %s", (b > a).reveal())  # 0
    dprint_ln(", expected output: 0")
    dprint("a < c: %s", (a < c).reveal())  # 0
    dprint_ln(", expected output: 0")
    dprint("a == c: %s", (a == c).reveal())  # 0
    dprint_ln(", expected output: 0")
    dprint("a > c: %s", (a > c).reveal())  # 1
    dprint_ln(", expected output: 1")
    dprint("c > a: %s", (c > a).reveal())  # 0
    dprint_ln(", expected output: 0")
    dprint("a < d: %s", (a < d).reveal())  # 1
    dprint_ln(", expected output: 1")
    dprint("d > a: %s", (d > a).reveal())  # 1
    dprint_ln(", expected output: 1")
    dprint("d == a: %s", (d == a).reveal())  # 0
    dprint_ln(", expected output: 0")
    dprint("c < b: %s", (c < b).reveal())  # 1
    dprint_ln(", expected output: 1")
    dprint("b < c: %s", (b < c).reveal())  # 0
    dprint_ln(", expected output: 0")
    dprint("b > c: %s", (b > c).reveal())  # 1
    dprint_ln(", expected output: 1")
    dprint("b == c: %s", (b == c).reveal())  # 0
    dprint_ln(", expected output: 0")

    # MemValues
    dprint("e < f: %s", (e < f).reveal())  # 0
    dprint_ln(", expected output: 0")
    dprint("f < e: %s", (f < e).reveal())  # 1
    dprint_ln(", expected output: 1")
    dprint("e == f: %s", (e == f).reveal())  # 0
    dprint_ln(", expected output: 0")

    # MemValues and basic types
    dprint("e < a: %s", (e < a).reveal())  # 1
    dprint_ln(", expected output: 1")
