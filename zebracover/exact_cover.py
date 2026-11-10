"""Exact cover with colours: Knuth's Algorithm C (TAOCP 7.2.2.1), dancing links, pure Python.

Primary items must be covered exactly once. Secondary items may be covered at most once, or by any number of
options that agree on its colour. That second rule is what turns a logic puzzle into a cover problem: give
``pos:red`` the colour of the house it is in, and every option that mentions ``pos:red`` must agree on where
red is.

    >>> ec = ExactCover(primary=["a", "b"], secondary=["x"])
    >>> ec.add(["a", ("x", 1)]); ec.add(["b", ("x", 1)]); ec.add(["b", ("x", 2)])
    >>> list(ec.solve())
    [[0, 1]]
"""
from __future__ import annotations

from typing import Hashable, Iterable, Iterator, Sequence

Item = Hashable


class ExactCover:
    def __init__(self, primary: Sequence[Item], secondary: Sequence[Item] = ()):
        self.primary, self.secondary = list(primary), list(secondary)
        self.index = {}
        for name in self.primary + self.secondary:
            if name in self.index:
                raise ValueError(f"duplicate item {name!r}")
            self.index[name] = len(self.index) + 1
        self.options: list[list[tuple[Item, Hashable | None]]] = []

    def add(self, option: Iterable[Item | tuple[Item, Hashable]]) -> int:
        """Add an option; a ``(name, colour)`` pair colours a secondary item. Returns the option's number."""
        parsed, seen = [], set()
        for entry in option:
            name, colour = entry if isinstance(entry, tuple) else (entry, None)
            if name not in self.index:
                raise KeyError(f"unknown item {name!r}")
            if name in seen:
                raise ValueError(f"item {name!r} twice in one option")
            if colour is not None and self.index[name] <= len(self.primary):
                raise ValueError(f"primary item {name!r} cannot be coloured")
            seen.add(name)
            parsed.append((name, colour))
        if not any(self.index[n] <= len(self.primary) for n, _ in parsed):
            raise ValueError("an option needs at least one primary item")
        self.options.append(parsed)
        return len(self.options) - 1

    def count(self, limit: int | None = None) -> int:
        n = 0
        for _ in self.solve(limit):
            n += 1
        return n

    def solve(self, limit: int | None = None) -> Iterator[list[int]]:
        """Yield solutions as sorted lists of option numbers, at most ``limit`` of them."""
        n1, n = len(self.primary), len(self.index)
        colour_ids: dict[Hashable, int] = {}
        # item headers 0..n (+ n+1 as the secondary list head), then nodes; spacer before every option
        llink, rlink = list(range(n + 2)), list(range(n + 2))
        for i in range(n1 + 1):
            llink[i], rlink[i] = (i - 1) % (n1 + 1), (i + 1) % (n1 + 1)
        head2 = n + 1
        sec = [head2] + list(range(n1 + 1, n + 1))
        for k, i in enumerate(sec):
            llink[i], rlink[i] = sec[k - 1], sec[(k + 1) % len(sec)]
        size = n + 2
        top, up, down, colour, length = list(range(size)), list(range(size)), list(range(size)), [0] * size, [0] * size
        owner = [-1] * size
        last_spacer = size
        top.append(0), up.append(0), down.append(0), colour.append(0), owner.append(-1)
        for number, opt in enumerate(self.options):
            first = len(top)
            for name, col in opt:
                i = self.index[name]
                node = len(top)
                top.append(i)
                up.append(up[i])
                down.append(i)
                down[up[i]] = node
                up[i] = node
                length[i] += 1
                colour.append(0 if col is None else colour_ids.setdefault(col, len(colour_ids) + 1))
                owner.append(number)
            down[last_spacer] = len(top) - 1
            spacer = len(top)
            top.append(-(number + 1)), up.append(first), down.append(0), colour.append(0), owner.append(-1)
            last_spacer = spacer
        length.extend([0] * (len(top) - len(length)))

        def hide(p):
            q = p + 1
            while q != p:
                x = top[q]
                if x <= 0:
                    q = up[q]
                elif colour[q] < 0:
                    q += 1
                else:
                    u, d = up[q], down[q]
                    down[u], up[d] = d, u
                    length[x] -= 1
                    q += 1

        def unhide(p):
            q = p - 1
            while q != p:
                x = top[q]
                if x <= 0:
                    q = down[q]
                elif colour[q] < 0:
                    q -= 1
                else:
                    u, d = up[q], down[q]
                    down[u] = up[d] = q
                    length[x] += 1
                    q -= 1

        def cover(i):
            p = down[i]
            while p != i:
                hide(p)
                p = down[p]
            left, right = llink[i], rlink[i]
            rlink[left], llink[right] = right, left

        def uncover(i):
            left, right = llink[i], rlink[i]
            rlink[left] = llink[right] = i
            p = up[i]
            while p != i:
                unhide(p)
                p = up[p]

        def purify(p):
            c, i = colour[p], top[p]
            q = down[i]
            while q != i:
                if colour[q] != c:
                    hide(q)
                elif q != p:  # as in Knuth's dlx2: p keeps its colour so uncommit can see it
                    colour[q] = -1
                q = down[q]

        def unpurify(p):
            c, i = colour[p], top[p]
            q = up[i]
            while q != i:
                if colour[q] < 0:
                    colour[q] = c
                else:
                    unhide(q)
                q = up[q]

        def commit(p):
            if colour[p] == 0:
                cover(top[p])
            elif colour[p] > 0:
                purify(p)

        def uncommit(p):
            if colour[p] == 0:
                uncover(top[p])
            elif colour[p] > 0:
                unpurify(p)

        def nodes_after(x):
            """The other nodes of x's option, in cyclic order starting after x."""
            out, q = [], x + 1
            while q != x:
                if top[q] <= 0:
                    q = up[q]
                    continue
                out.append(q)
                q += 1
            return out

        found = 0
        chosen: list[int] = []

        def search():
            nonlocal found
            if rlink[0] == 0:
                found += 1
                yield sorted(owner[x] for x in chosen)
                return
            best, best_len, i = None, None, rlink[0]
            while i != 0:
                if best_len is None or length[i] < best_len:
                    best, best_len = i, length[i]
                    if best_len == 0:
                        break
                i = rlink[i]
            if best_len == 0:
                return
            cover(best)
            x = down[best]
            while x != best:
                others = nodes_after(x)
                for p in others:
                    commit(p)
                chosen.append(x)
                yield from search()
                chosen.pop()
                for p in reversed(others):
                    uncommit(p)
                if limit is not None and found >= limit:
                    break
                x = down[x]
            uncover(best)

        for solution in search():
            yield solution
            if limit is not None and found >= limit:
                return
