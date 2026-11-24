"""Node-graph execution engine (headless, no GUI imports).

A `Graph` is nodes + links; an `Executor` runs it in dependency order, caching the outputs of nodes whose
type, parameters and upstream have not changed. Node types live in `REGISTRY` (see nodes.py for the built-in
ones). Studio shows this as a visual node editor and calls `Executor.run` on a worker thread.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import traceback
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Callable

PORT_TYPES = ("robot", "policy", "motion", "number", "int", "bool", "text", "choice", "code", "series",
              "points", "image", "table", "any")
PARAM_TYPES = ("number", "int", "bool", "text", "choice", "code")

CATEGORY_COLORS = {
    "Robot": "#2b6cb0", "World": "#2f855a", "Simulate": "#c05621", "Sensors": "#0987a0", "Train": "#b83280",
    "Policy": "#6b46c1", "Design": "#b7791f", "Script": "#4a5568", "Value": "#718096", "Output": "#c53030",
}


class GraphError(Exception):
    """A graph cannot be built or a node cannot run as wired (bad link, missing input, bad parameter)."""


class Cancelled(Exception):
    """Raised inside a node when the run's cancel event is set."""


@dataclass
class Port:
    name: str
    type: str
    default: object = None
    choices: list | None = None     # for "choice"
    min: float | None = None
    max: float | None = None
    doc: str = ""
    optional: bool = False          # input sockets only: may stay unwired with default None (passed as None)


@dataclass
class NodeType:
    id: str
    title: str
    category: str
    inputs: list[Port]
    outputs: list[Port]
    params: list[Port]
    fn: Callable                    # fn(ctx, **inputs, **params) -> dict {output_name: value}
    doc: str = ""
    volatile: bool = False          # never cached (reads live state, has side effects)
    color: str = "#4a5568"

    def input(self, name: str) -> Port | None:
        return next((p for p in self.inputs if p.name == name), None)

    def output(self, name: str) -> Port | None:
        return next((p for p in self.outputs if p.name == name), None)

    def param(self, name: str) -> Port | None:
        return next((p for p in self.params if p.name == name), None)

    def default_params(self) -> dict:
        return {p.name: _copy_default(p.default) for p in self.params}


REGISTRY: dict[str, NodeType] = {}


def _copy_default(v):
    return list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v


def register(nt: NodeType) -> None:
    """Add a node type to REGISTRY (replacing one with the same id)."""
    names = [p.name for p in nt.inputs] + [p.name for p in nt.params]
    if len(set(names)) != len(names) or "ctx" in names:
        raise GraphError(f"{nt.id}: input and parameter names must be unique and not 'ctx' ({names})")
    outs = [p.name for p in nt.outputs]
    if len(set(outs)) != len(outs):
        raise GraphError(f"{nt.id}: duplicate output names ({outs})")
    for p in [*nt.inputs, *nt.outputs, *nt.params]:
        if p.type not in PORT_TYPES:
            raise GraphError(f"{nt.id}: port {p.name!r} has unknown type {p.type!r}")
    for p in nt.params:
        if p.type not in PARAM_TYPES:
            raise GraphError(f"{nt.id}: parameter {p.name!r} must be one of {PARAM_TYPES}, not {p.type!r}")
    REGISTRY[nt.id] = nt


def node(id, title, category, inputs=(), outputs=(), params=(), doc="", volatile=False, color=None):
    """Decorator: register `fn(ctx, **inputs, **params) -> dict` as a node type."""
    def deco(fn):
        register(NodeType(id=id, title=title, category=category, inputs=list(inputs), outputs=list(outputs),
                          params=list(params), fn=fn, doc=doc or (fn.__doc__ or "").strip().split("\n")[0],
                          volatile=volatile, color=color or CATEGORY_COLORS.get(category, "#4a5568")))
        return fn
    return deco


def compatible(out_type: str, in_type: str) -> bool:
    """Can an output of `out_type` feed an input of `in_type`?"""
    return out_type == in_type or "any" in (out_type, in_type) or (out_type == "int" and in_type == "number")


def coerce(port: Port, value):
    """A parameter value as the type its port declares (JSON files and text fields give loose types)."""
    t = port.type
    try:
        if value is None:
            return None
        if t == "number":
            v = float(value)
        elif t == "int":
            v = int(round(float(value)))
        elif t == "bool":
            return value.strip().lower() in ("1", "true", "yes", "on") if isinstance(value, str) else bool(value)
        elif t in ("text", "code", "choice"):
            v = str(value)
            if t == "choice" and port.choices and v not in [str(c) for c in port.choices]:
                raise GraphError(f"{port.name} must be one of {port.choices}, not {v!r}")
            return v
        else:
            return value
    except (TypeError, ValueError):
        raise GraphError(f"{port.name}: {value!r} is not a valid {t}") from None
    if port.min is not None:
        v = max(type(v)(port.min), v)
    if port.max is not None:
        v = min(type(v)(port.max), v)
    return v


@dataclass
class NodeInstance:
    id: str
    type: str
    params: dict = field(default_factory=dict)
    pos: tuple[float, float] = (0, 0)
    title: str | None = None


@dataclass
class Link:
    src: str
    src_port: str
    dst: str
    dst_port: str


class Graph:
    def __init__(self):
        self.nodes: dict[str, NodeInstance] = {}
        self.links: list[Link] = []

    # ---- editing ---------------------------------------------------------
    def add_node(self, type_id, pos=(0, 0), params=None, id=None, title=None) -> NodeInstance:
        nt = REGISTRY.get(type_id)
        if nt is None:
            raise GraphError(f"unknown node type {type_id!r}")
        if id is None:
            base, k = type_id.replace(".", "_"), 1
            while f"{base}_{k}" in self.nodes:
                k += 1
            id = f"{base}_{k}"
        elif id in self.nodes:
            raise GraphError(f"node id {id!r} already exists")
        p = nt.default_params()
        for k, v in (params or {}).items():
            if nt.param(k) is None:
                raise GraphError(f"{type_id} has no parameter {k!r}")
            p[k] = v
        n = NodeInstance(id=str(id), type=type_id, params=p, pos=(float(pos[0]), float(pos[1])), title=title)
        self.nodes[n.id] = n
        return n

    def remove_node(self, node_id) -> None:
        if node_id not in self.nodes:
            raise GraphError(f"no node {node_id!r}")
        del self.nodes[node_id]
        self.links = [l for l in self.links if l.src != node_id and l.dst != node_id]

    def can_connect(self, src, src_port, dst, dst_port) -> tuple[bool, str]:
        a, b = self.nodes.get(src), self.nodes.get(dst)
        if a is None or b is None:
            return False, f"no node {src if a is None else dst!r}"
        ta, tb = REGISTRY.get(a.type), REGISTRY.get(b.type)
        if ta is None or tb is None:
            return False, f"unknown node type {(a if ta is None else b).type!r}"
        po, pi = ta.output(src_port), tb.input(dst_port)
        if po is None:
            return False, f"{ta.title} has no output {src_port!r}"
        if pi is None:
            return False, f"{tb.title} has no input {dst_port!r}"
        if not compatible(po.type, pi.type):
            return False, f"cannot connect {po.type} to {pi.type}"
        if src == dst or src in self._descendants(dst):
            return False, "that link would make a loop"
        return True, ""

    def connect(self, src, src_port, dst, dst_port) -> Link:
        ok, why = self.can_connect(src, src_port, dst, dst_port)
        if not ok:
            raise GraphError(why)
        self.links = [l for l in self.links if not (l.dst == dst and l.dst_port == dst_port)]
        link = Link(src, src_port, dst, dst_port)
        self.links.append(link)
        return link

    def disconnect(self, dst, dst_port) -> None:
        self.links = [l for l in self.links if not (l.dst == dst and l.dst_port == dst_port)]

    # ---- structure -------------------------------------------------------
    def inputs_of(self, node_id) -> dict[str, Link]:
        return {l.dst_port: l for l in self.links if l.dst == node_id}

    def _descendants(self, node_id) -> set[str]:
        out, stack = set(), [node_id]
        while stack:
            n = stack.pop()
            for l in self.links:
                if l.src == n and l.dst not in out:
                    out.add(l.dst)
                    stack.append(l.dst)
        return out

    def _ancestors(self, node_ids) -> set[str]:
        out, stack = set(), list(node_ids)
        while stack:
            n = stack.pop()
            if n in out:
                continue
            out.add(n)
            stack.extend(l.src for l in self.links if l.dst == n)
        return out

    def topo_order(self, targets=None) -> list[str]:
        """Node ids in execution order (ties in insertion order); only `targets` and their ancestors if given."""
        if targets is not None:
            targets = [targets] if isinstance(targets, str) else list(targets)
            for t in targets:
                if t not in self.nodes:
                    raise GraphError(f"no node {t!r}")
            keep = self._ancestors(targets)
        else:
            keep = set(self.nodes)
        deps = {n: {l.src for l in self.links if l.dst == n and l.src in keep} for n in self.nodes if n in keep}
        order, done = [], set()
        while len(order) < len(deps):
            ready = [n for n in deps if n not in done and deps[n] <= done]
            if not ready:
                raise GraphError("the graph has a loop")
            order.append(ready[0])
            done.add(ready[0])
        return order

    def validate(self) -> list[str]:
        """Human-readable problems; an empty list means the graph can run."""
        out = []
        for n in self.nodes.values():
            nt = REGISTRY.get(n.type)
            name = n.title or (nt.title if nt else n.type)
            if nt is None:
                out.append(f"{n.id}: unknown node type {n.type!r}")
                continue
            wired = self.inputs_of(n.id)
            for p in nt.inputs:
                if p.name not in wired and p.default is None and not p.optional:
                    out.append(f"{name} ({n.id}): input '{p.name}' is not connected")
            for p in nt.params:
                try:
                    coerce(p, n.params.get(p.name, p.default))
                except GraphError as e:
                    out.append(f"{name} ({n.id}): {e}")
        for l in self.links:
            a, b = self.nodes.get(l.src), self.nodes.get(l.dst)
            if a is None or b is None:
                out.append(f"link {l.src}.{l.src_port} -> {l.dst}.{l.dst_port}: missing node")
                continue
            ta, tb = REGISTRY.get(a.type), REGISTRY.get(b.type)
            if ta is None or tb is None:
                continue
            po, pi = ta.output(l.src_port), tb.input(l.dst_port)
            if po is None or pi is None:
                out.append(f"link {l.src}.{l.src_port} -> {l.dst}.{l.dst_port}: no such port")
            elif not compatible(po.type, pi.type):
                out.append(f"link {l.src}.{l.src_port} -> {l.dst}.{l.dst_port}: {po.type} does not fit {pi.type}")
        try:
            self.topo_order()
        except GraphError as e:
            out.append(str(e))
        return out

    # ---- persistence -----------------------------------------------------
    def to_dict(self) -> dict:
        return {"version": 1,
                "nodes": [{"id": n.id, "type": n.type, "params": dict(n.params), "pos": [n.pos[0], n.pos[1]],
                           "title": n.title} for n in self.nodes.values()],
                "links": [{"src": l.src, "src_port": l.src_port, "dst": l.dst, "dst_port": l.dst_port}
                          for l in self.links]}

    @classmethod
    def from_dict(cls, d: dict) -> "Graph":
        """Rebuild a graph. Unknown node types and stale links are kept as they are (see validate())."""
        if int(d.get("version", 1)) != 1:
            raise GraphError(f"unsupported graph version {d.get('version')!r}")
        g = cls()
        for nd in d.get("nodes", []):
            nt = REGISTRY.get(nd["type"])
            params = nt.default_params() if nt else {}
            params.update(nd.get("params") or {})
            pos = nd.get("pos") or (0, 0)
            g.nodes[str(nd["id"])] = NodeInstance(id=str(nd["id"]), type=nd["type"], params=params,
                                                  pos=(float(pos[0]), float(pos[1])), title=nd.get("title"))
        for ld in d.get("links", []):
            g.links.append(Link(str(ld["src"]), ld["src_port"], str(ld["dst"]), ld["dst_port"]))
        return g

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Graph":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---- execution -----------------------------------------------------------------------------------------
def summarize(value) -> str:
    """A short text for any wire value (shown on node bodies)."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, str):
        s = value.strip()
        first = s.split("\n")[0]
        return first[:77] + "..." if len(first) > 80 or "\n" in s and len(first) > 77 else (
            first + (" ..." if "\n" in s else ""))
    if hasattr(value, "summary") and callable(value.summary):
        return str(value.summary())
    if np is not None and isinstance(value, np.generic):
        return summarize(value.item())
    if np is not None and isinstance(value, np.ndarray):
        if value.ndim == 3 and value.shape[2] in (1, 3, 4) and value.dtype == np.uint8:
            return f"image {value.shape[1]}x{value.shape[0]}"
        if value.ndim == 2 and value.shape[1] == 3:
            return f"{len(value)} points"
        return f"array {'x'.join(str(s) for s in value.shape)} {value.dtype}"
    if isinstance(value, dict):
        if "x" in value and isinstance(value.get("y"), dict):
            return f"series '{value.get('title', '')}': {len(value['y'])} line(s), {len(value['x'])} points"
        if "path" in value and "name" in value:
            return f"policy {value['name']} ({value.get('kind', '?')})"
        return "{" + ", ".join(str(k) for k in list(value)[:6]) + (", ..." if len(value) > 6 else "") + "}"
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(r, dict) for r in value):
            return f"table: {len(value)} rows x {len(value[0])} columns"
        s = repr(value)
        return s if len(s) <= 80 else f"list of {len(value)}"
    s = repr(value)
    return s if len(s) <= 80 else s[:77] + "..."


def jsonable(v):
    """Wire values -> plain JSON types (numpy arrays to lists, dataclasses to dicts, NaN/inf to None)."""
    import math
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [jsonable(x) for x in v]
    if hasattr(v, "to_dict") and callable(v.to_dict):
        return jsonable(v.to_dict())
    if is_dataclass(v) and not isinstance(v, type):
        from dataclasses import asdict
        return jsonable(asdict(v))
    if hasattr(v, "tolist"):
        return jsonable(v.tolist())
    if isinstance(v, Path):
        return str(v)
    return repr(v)


class Context:
    """What a node function gets as `ctx`: the session, progress / log reporting and cancellation."""

    def __init__(self, executor: "Executor", node_id: str, emit, cancel: threading.Event | None):
        self._executor, self._emit, self._cancel = executor, emit, cancel
        self.node_id = node_id

    @property
    def session(self):
        """The live SimSession, or None when the executor has none yet (see need_session)."""
        return self._executor.session

    def need_session(self):
        """The live SimSession; a private one is created on first use when the executor was given none."""
        return self._executor.need_session()

    def check_cancel(self) -> None:
        if self._cancel is not None and self._cancel.is_set():
            raise Cancelled()

    def progress(self, frac: float, message: str = "", **data) -> None:
        self.check_cancel()
        self._emit("progress", self.node_id, {"frac": float(min(1.0, max(0.0, frac))), "message": message, **data})

    def log(self, message: str) -> None:
        self._emit("log", self.node_id, {"message": str(message)})

    def sleep(self, seconds: float) -> None:
        """Wait in short slices, raising Cancelled as soon as the run is cancelled."""
        end = time.monotonic() + max(0.0, seconds)
        while True:
            self.check_cancel()
            left = end - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(0.05, left))


class Executor:
    """Runs graphs. Keeps a cache (node id -> (signature, outputs)) between runs."""

    def __init__(self, session=None):
        self.session = session
        self._own_session = None
        self.cache: dict[str, tuple[str, dict]] = {}
        self.results: dict[str, dict] = {}     # outputs of the current / last run, filled in as nodes finish
        self._run_lock = threading.Lock()
        self._session_lock = threading.Lock()

    def need_session(self):
        with self._session_lock:
            if self.session is None:
                from ..server.session import SimSession
                self._own_session = self.session = SimSession()
            return self.session

    def clear_cache(self) -> None:
        self.cache.clear()

    def close(self) -> None:
        """Shut down the session this executor created for itself (a session passed in is left alone)."""
        with self._session_lock:
            s, self._own_session = self._own_session, None
            if s is not None:
                if self.session is s:
                    self.session = None
                s.running = False
                try:
                    s.pilot.stop(timeout=1.0)
                except Exception:  # noqa: BLE001
                    pass
                s.shutdown()

    # ---- signatures ------------------------------------------------------
    @staticmethod
    def _params(nt: NodeType, inst: NodeInstance) -> dict:
        return {p.name: coerce(p, inst.params.get(p.name, p.default)) for p in nt.params}

    def signatures(self, graph: Graph, order: list[str] | None = None) -> dict[str, str | None]:
        """Cache key per node: a hash of its type, parameters and upstream keys. None = always re-run
        (volatile nodes, everything downstream of them, and nodes that cannot be keyed)."""
        sigs: dict[str, str | None] = {}
        for nid in order if order is not None else graph.topo_order():
            inst = graph.nodes[nid]
            nt = REGISTRY.get(inst.type)
            sig = None
            if nt is not None and not nt.volatile:
                ups = sorted((l.dst_port, sigs.get(l.src), l.src_port) for l in graph.links if l.dst == nid)
                if all(u[1] is not None for u in ups):
                    try:
                        blob = json.dumps([nt.id, self._params(nt, inst), ups], sort_keys=True, default=repr)
                        sig = hashlib.sha1(blob.encode()).hexdigest()
                    except GraphError:
                        sig = None
            sigs[nid] = sig
        return sigs

    # ---- running ---------------------------------------------------------
    def run(self, graph: Graph, targets=None, on_event=None, cancel: threading.Event | None = None) -> dict[str, dict]:
        """Run `targets` and what they depend on (everything if None). Blocking. Returns node id -> outputs
        for the nodes that completed; a failing node stops the run without raising (see the "error" event)."""
        def emit(kind, node_id, payload):
            if on_event is not None:
                try:
                    on_event(kind, node_id, payload)
                except Exception:  # noqa: BLE001 - a broken listener must not break the run
                    traceback.print_exc()

        with self._run_lock:
            t_run = time.perf_counter()
            results: dict[str, dict] = {}
            self.results = results
            ok, cancelled = False, False
            try:
                try:
                    order = graph.topo_order(targets)
                    sigs = self.signatures(graph, order)
                except GraphError as e:
                    emit("error", None, {"message": str(e), "traceback": ""})
                    return results
                for nid in order:
                    if cancel is not None and cancel.is_set():
                        cancelled = True
                        return results
                    inst = graph.nodes[nid]
                    nt = REGISTRY.get(inst.type)
                    sig = sigs[nid]
                    hit = self.cache.get(nid)
                    if sig is not None and hit is not None and hit[0] == sig:
                        results[nid] = hit[1]
                        emit("cached", nid, {"outputs": {k: summarize(v) for k, v in hit[1].items()}})
                        continue
                    emit("start", nid, {"title": inst.title or (nt.title if nt else inst.type)})
                    t0 = time.perf_counter()
                    try:
                        if nt is None:
                            raise GraphError(f"unknown node type {inst.type!r}")
                        kwargs = self._params(nt, inst)
                        wired = graph.inputs_of(nid)
                        for p in nt.inputs:
                            link = wired.get(p.name)
                            if link is not None:
                                kwargs[p.name] = results[link.src].get(link.src_port)
                            elif p.default is None and not p.optional:
                                raise GraphError(f"input '{p.name}' is not connected")
                            else:
                                kwargs[p.name] = _copy_default(p.default)
                        out = nt.fn(Context(self, nid, emit, cancel), **kwargs)
                        if not isinstance(out, dict):
                            if len(nt.outputs) == 1:
                                out = {nt.outputs[0].name: out}
                            elif out is None:
                                out = {}
                            else:
                                raise GraphError(f"{nt.id} must return a dict of outputs, not {type(out).__name__}")
                        out = {p.name: out.get(p.name) for p in nt.outputs}
                    except Cancelled:
                        cancelled = True
                        self.cache.pop(nid, None)
                        return results
                    except Exception as e:  # noqa: BLE001 - reported to the editor, the run stops here
                        self.cache.pop(nid, None)
                        msg = str(e) if isinstance(e, GraphError) else f"{type(e).__name__}: {e}"
                        emit("error", nid, {"message": msg, "traceback": traceback.format_exc()})
                        return results
                    results[nid] = out
                    if sig is not None:
                        self.cache[nid] = (sig, out)
                    else:
                        self.cache.pop(nid, None)
                    emit("done", nid, {"outputs": {k: summarize(v) for k, v in out.items()},
                                       "seconds": time.perf_counter() - t0})
                ok = True
                return results
            finally:
                emit("finished", None, {"ok": ok, "cancelled": cancelled, "seconds": time.perf_counter() - t_run})
