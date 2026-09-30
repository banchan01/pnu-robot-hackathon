"""Behavior Tree 핵심 (강의 노트북 4장 C절 코드 기반, 담당 D).

노트북 제공: Status, Node, Composite, Selector, Sequence, Parallel, Blackboard, BehaviorTree
추가 구현: ReactiveSequence, ReactiveSelector, Recovery, Condition, Action, Inverter
"""
from enum import Enum


class Status(Enum):
    SUCCESS = 1
    FAILURE = 2
    RUNNING = 3


class Node:
    def __init__(self, name):
        self.name = name
        self.status = None

    def tick(self):
        raise NotImplementedError

    def reset(self):
        pass

    def _ret(self, s):
        self.status = s
        return s


class Composite(Node):
    def __init__(self, name, children=None):
        super().__init__(name)
        self.children = list(children) if children else []

    def add_child(self, child):
        self.children.append(child)
        return self

    def reset(self):
        for c in self.children:
            c.reset()


class Selector(Composite):
    """메모리 기반: Running 자식부터 재개."""
    def __init__(self, name, children=None):
        super().__init__(name, children)
        self.current_child = 0

    def tick(self):
        while self.current_child < len(self.children):
            s = self.children[self.current_child].tick()
            if s == Status.FAILURE:
                self.current_child += 1
            elif s == Status.RUNNING:
                return self._ret(Status.RUNNING)
            else:
                self.current_child = 0
                return self._ret(Status.SUCCESS)
        self.current_child = 0
        return self._ret(Status.FAILURE)

    def reset(self):
        super().reset()
        self.current_child = 0


class Sequence(Composite):
    """메모리 기반: Running 자식부터 재개."""
    def __init__(self, name, children=None):
        super().__init__(name, children)
        self.current_child = 0

    def tick(self):
        while self.current_child < len(self.children):
            s = self.children[self.current_child].tick()
            if s == Status.SUCCESS:
                self.current_child += 1
            elif s == Status.RUNNING:
                return self._ret(Status.RUNNING)
            else:
                self.reset()
                return self._ret(Status.FAILURE)
        self.reset()
        return self._ret(Status.SUCCESS)

    def reset(self):
        super().reset()
        self.current_child = 0


class Parallel(Composite):
    def __init__(self, name, children=None, success_count=1):
        super().__init__(name, children)
        self.success_count = success_count

    def tick(self):
        statuses = [c.tick() for c in self.children]
        if Status.RUNNING in statuses:
            return self._ret(Status.RUNNING)
        if statuses.count(Status.SUCCESS) >= self.success_count:
            return self._ret(Status.SUCCESS)
        return self._ret(Status.FAILURE)


class ReactiveSequence(Composite):
    """매 tick 첫 자식부터 다시 평가. 앞 조건이 깨지면 뒤 자식은 실행되지 않는다."""
    def tick(self):
        for i, c in enumerate(self.children):
            s = c.tick()
            if s == Status.FAILURE:
                for later in self.children[i + 1:]:
                    later.reset()
                return self._ret(Status.FAILURE)
            if s == Status.RUNNING:
                for later in self.children[i + 1:]:
                    later.reset()
                return self._ret(Status.RUNNING)
        return self._ret(Status.SUCCESS)


class ReactiveSelector(Composite):
    """매 tick 첫 자식부터 다시 평가. 우선순위 높은 자식이 살아나면 뒤 자식은 reset."""
    def tick(self):
        for i, c in enumerate(self.children):
            s = c.tick()
            if s == Status.SUCCESS or s == Status.RUNNING:
                for later in self.children[i + 1:]:
                    later.reset()
                return self._ret(s)
        return self._ret(Status.FAILURE)


class Recovery(Composite):
    """자식 2개: [주 행동, 복구 행동]. 주 행동 실패 시 복구 행동을 실행하고 다시 시도."""
    def __init__(self, name, main, recovery, max_retries=3):
        super().__init__(name, [main, recovery])
        self.max_retries = max_retries
        self.retries = 0
        self.in_recovery = False

    def tick(self):
        main, rec = self.children
        if self.in_recovery:
            s = rec.tick()
            if s == Status.RUNNING:
                return self._ret(Status.RUNNING)
            self.in_recovery = False
            rec.reset()
            if s == Status.FAILURE:
                return self._ret(Status.FAILURE)
            return self._ret(Status.RUNNING)     # 다음 tick에 주 행동 재시도
        s = main.tick()
        if s == Status.SUCCESS:
            self.retries = 0
            return self._ret(Status.SUCCESS)
        if s == Status.RUNNING:
            return self._ret(Status.RUNNING)
        self.retries += 1
        if self.retries > self.max_retries:
            self.retries = 0
            return self._ret(Status.FAILURE)
        main.reset()
        self.in_recovery = True
        return self._ret(Status.RUNNING)

    def reset(self):
        super().reset()
        self.retries = 0
        self.in_recovery = False


class Condition(Node):
    def __init__(self, name, fn):
        super().__init__(name)
        self.fn = fn

    def tick(self):
        return self._ret(Status.SUCCESS if self.fn() else Status.FAILURE)


class Action(Node):
    """fn()이 Status를 돌려주는 잎 노드. reset_fn은 선택."""
    def __init__(self, name, fn, reset_fn=None):
        super().__init__(name)
        self.fn = fn
        self.reset_fn = reset_fn

    def tick(self):
        return self._ret(self.fn())

    def reset(self):
        if self.reset_fn:
            self.reset_fn()


class Inverter(Node):
    def __init__(self, name, child):
        super().__init__(name)
        self.child = child

    def tick(self):
        s = self.child.tick()
        if s == Status.SUCCESS:
            return self._ret(Status.FAILURE)
        if s == Status.FAILURE:
            return self._ret(Status.SUCCESS)
        return self._ret(Status.RUNNING)

    def reset(self):
        self.child.reset()


class Blackboard:
    def __init__(self):
        self.data = {}

    def set(self, key, value):
        self.data[key] = value

    def get(self, key, default=None):
        return self.data.get(key, default)

    def has(self, key):
        return key in self.data

    def remove(self, key):
        self.data.pop(key, None)

    def clear(self):
        self.data.clear()


class BehaviorTree:
    def __init__(self, root):
        self.root = root

    def tick(self):
        return self.root.tick()

    def reset(self):
        self.root.reset()


def describe(node, depth=0, out=None):
    """트리 구조를 텍스트로 (README와 로그용)."""
    if out is None:
        out = []
    mark = ""
    if node.status is not None:
        mark = {Status.SUCCESS: " [S]", Status.FAILURE: " [F]", Status.RUNNING: " [R]"}[node.status]
    out.append("  " * depth + f"{node.__class__.__name__}: {node.name}{mark}")
    for c in getattr(node, "children", []) or ([node.child] if hasattr(node, "child") else []):
        describe(c, depth + 1, out)
    return out
