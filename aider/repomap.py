import colorsys
import math
import os
import random
import shutil
import sqlite3
import sys
import time
import warnings
from collections import Counter, defaultdict, namedtuple, OrderedDict
from importlib import resources
from pathlib import Path

from diskcache import Cache
from grep_ast import TreeContext, filename_to_lang
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from tqdm import tqdm

from aider.dump import dump
from aider.special import filter_important_files
from aider.waiting import Spinner
import requests
import urllib.parse
import socket
import subprocess
import atexit

# tree_sitter is throwing a FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)
from grep_ast.tsl import USING_TSL_PACK, get_language, get_parser  # noqa: E402


class LSPGatewayClient:
    def __init__(self, timeout=5, io=None, verbose=False):
        self.timeout = timeout
        self._languages = None
        self._rid = 0
        self._proc = None
        self._port = None
        self.base_url = None
        self.io = io
        self.verbose = verbose
        self.stats = {"calls": 0, "time_total": 0.0, "by_method": {}}
        try:
            self._start_server()
        except Exception:
            self._proc = None
            self.base_url = None
        atexit.register(self.close)

    def _next_id(self):
        self._rid += 1
        return self._rid

    def _jsonrpc(self, method, params):
        if not self.base_url:
            raise RuntimeError("LSP gateway unavailable")
        url = self.base_url.rstrip("/") + "/jsonrpc"
        payload = {"jsonrpc": "2.0", "id": self._next_id(), "method": method, "params": params}
        t0 = time.time()
        r = requests.post(url, json=payload, timeout=self.timeout)
        t1 = time.time()
        dt = (t1 - t0) * 1000.0
        r.raise_for_status()
        data = r.json()
        if "error" in data and data["error"]:
            raise RuntimeError(str(data["error"]))
        result = data.get("result")
        self.stats["calls"] += 1
        self.stats["time_total"] += dt
        bm = self.stats["by_method"].setdefault(method, {"count": 0, "time_total": 0.0})
        bm["count"] += 1
        bm["time_total"] += dt
        if self.verbose and self.io:
            uri = None
            if isinstance(params, dict):
                td = params.get("textDocument") or {}
                uri = td.get("uri")
            size = 0
            try:
                size = len(result) if isinstance(result, list) else (1 if result else 0)
            except Exception:
                size = 0
            self.io.tool_output(
                f"LSP {method} uri={uri or '-'} took {dt:.1f}ms, items={size}"
            )
        return result

    def _fetch_languages(self):
        if not self.base_url:
            return {"languages": [], "extensions": {}}
        url = self.base_url.rstrip("/") + "/languages"
        t0 = time.time()
        r = requests.get(url, timeout=self.timeout)
        t1 = time.time()
        dt = (t1 - t0) * 1000.0
        r.raise_for_status()
        res = r.json()
        if self.verbose and self.io:
            langs = len(res.get("languages", []))
            self.io.tool_output(
                f"LSP /languages {self.base_url} took {dt:.1f}ms, count={langs}"
            )
        return res

    def languages(self):
        if self._languages is None:
            try:
                self._languages = self._fetch_languages()
            except Exception:
                self._languages = {"languages": [], "extensions": {}}
        return self._languages

    def is_supported_file(self, path):
        if not self.base_url:
            return False
        exts = self.languages().get("extensions", {})
        suffix = Path(path).suffix.lower()
        for v in exts.values():
            if suffix in [e.lower() for e in v]:
                return True
        return False

    def document_symbols(self, uri):
        res = self._jsonrpc("textDocument/documentSymbol", {"textDocument": {"uri": uri}})
        if not isinstance(res, list):
            return []
        return res

    def definition(self, uri, position):
        return self._jsonrpc(
            "textDocument/definition",
            {"textDocument": {"uri": uri}, "position": position},
        )

    def references(self, uri, position):
        return self._jsonrpc(
            "textDocument/references",
            {
                "textDocument": {"uri": uri},
                "position": position,
                "context": {"includeDeclaration": False},
            },
        )
    def declaration(self, uri, position):
        return self._jsonrpc(
            "textDocument/declaration",
            {"textDocument": {"uri": uri}, "position": position},
        )

    def _start_server(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        exe = shutil.which("lsp-gateway")
        if not exe:
            raise FileNotFoundError("lsp-gateway executable not found")
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self._port = sock.getsockname()[1]
        sock.close()

        args = [exe, "server", "--port", str(self._port)]
        creationflags = 0
        preexec_fn = None
        try:
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                preexec_fn = os.setsid
        except Exception:
            pass
        t0 = time.time()
        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            preexec_fn=preexec_fn,
            creationflags=creationflags,
        )
        self.base_url = f"http://127.0.0.1:{self._port}"
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                _ = self._fetch_languages()
                break
            except Exception:
                time.sleep(0.2)
        if self.verbose and self.io:
            dt = (time.time() - t0) * 1000.0
            self.io.tool_output(f"LSP server started at {self.base_url} in {dt:.1f}ms")

    def format_stats(self):
        ms_total = self.stats.get("time_total", 0.0)
        calls = self.stats.get("calls", 0)
        lines = [f"LSP stats: calls={calls}, time={ms_total:.1f}ms"]
        for m, s in sorted(self.stats.get("by_method", {}).items()):
            lines.append(f"  {m}: {s['count']} calls, {s['time_total']:.1f}ms")
        return "\n".join(lines)

    def close(self):
        try:
            if self._proc and self._proc.poll() is None:
                if os.name == "nt":
                    self._proc.terminate()
                else:
                    try:
                        os.killpg(self._proc.pid, 15)
                    except Exception:
                        self._proc.terminate()
        except Exception:
            pass


SYMBOL_KIND_NAMES = {
    "file": 1,
    "module": 2,
    "namespace": 3,
    "package": 4,
    "class": 5,
    "method": 6,
    "property": 7,
    "field": 8,
    "constructor": 9,
    "enum": 10,
    "interface": 11,
    "function": 12,
    "variable": 13,
    "constant": 14,
    "string": 15,
    "number": 16,
    "boolean": 17,
    "array": 18,
    "object": 19,
    "key": 20,
    "null": 21,
    "enummember": 22,
    "struct": 23,
    "event": 24,
    "operator": 25,
    "typeparameter": 26,
}

EXCLUDED_SYMBOL_KINDS = {1, 2, 3, 4, 13, 14, 15, 16, 17, 18, 19, 20, 21}


def _kind_code(kind):
    if isinstance(kind, int):
        return kind
    if isinstance(kind, str):
        return SYMBOL_KIND_NAMES.get(kind.strip().lower())
    return None


def flatten_document_symbols(symbols):
    out = []
    for s in symbols or []:
        if not isinstance(s, dict):
            continue
        k = _kind_code(s.get("kind"))
        # Always traverse children; only skip appending current if excluded
        if "children" in s and s.get("children"):
            out.extend(flatten_document_symbols(s.get("children") or []))

        if "location" in s:
            loc = s.get("location", {})
            if k in EXCLUDED_SYMBOL_KINDS:
                continue
            out.append(
                {
                    "name": s.get("name"),
                    "kind": s.get("kind"),
                    "range": loc.get("range"),
                    "selectionRange": loc.get("range"),
                }
            )
            continue

        if k in EXCLUDED_SYMBOL_KINDS:
            continue
        out.append(s)
    return out


def build_ident_loc(rel_fname, start_pos):
    line = start_pos.get("line", 0)
    ch = start_pos.get("character", 0)
    return f"{rel_fname}@{line}:{ch}"


def path_to_uri(path):
    try:
        return Path(path).resolve().as_uri()
    except Exception:
        p = os.path.abspath(path)
        return "file://" + urllib.parse.quote(p)


def uri_to_path(uri):
    try:
        if uri.startswith("file://"):
            parsed = urllib.parse.urlparse(uri)
            return urllib.parse.unquote(parsed.path)
        return uri
    except Exception:
        return uri

Tag = namedtuple("Tag", "rel_fname fname line name kind".split())


SQLITE_ERRORS = (sqlite3.OperationalError, sqlite3.DatabaseError, OSError)


CACHE_VERSION = 3
if USING_TSL_PACK:
    CACHE_VERSION = 4

UPDATING_REPO_MAP_MESSAGE = "Updating repo map"


class RepoMap:
    TAGS_CACHE_DIR = f".aider.tags.cache.v{CACHE_VERSION}"

    warned_files = set()

    def __init__(
        self,
        map_tokens=1024,
        root=None,
        main_model=None,
        io=None,
        repo_content_prefix=None,
        verbose=False,
        max_context_window=None,
        map_mul_no_files=8,
        refresh="auto",
    ):
        self.io = io
        self.verbose = verbose
        self.refresh = refresh

        if not root:
            root = os.getcwd()
        self.root = root
        self.ignore_spec = self._load_ignore_spec()

        self.load_tags_cache()
        self.cache_threshold = 0.95

        self.max_map_tokens = map_tokens
        self.map_mul_no_files = map_mul_no_files
        self.max_context_window = max_context_window

        self.repo_content_prefix = repo_content_prefix

        self.main_model = main_model

        self.tree_cache = {}
        self.tree_context_cache = {}
        self.map_cache = {}
        self.map_processing_time = 0
        self.last_map = None

        self.lsp = LSPGatewayClient(io=self.io, verbose=self.verbose)

        # Mapping: rel_fname -> { line_number -> rank }
        self.rank_by_file_line = {}

        if self.verbose:
            self.io.tool_output(
                f"RepoMap initialized with map_mul_no_files: {self.map_mul_no_files}"
            )

        self.LSP_MEM_CACHE = {}

    # ----------------------
    # LSP helper utilities
    # ----------------------
    def _abs_from_uri(self, uri):
        try:
            if not uri:
                return None
            return Path(uri_to_path(uri)).resolve().as_posix()
        except Exception:
            return None

    def _cache_get(self, key):
        try:
            return self.TAGS_CACHE.get(key)
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)
            return self.TAGS_CACHE.get(key)

    def _cache_set(self, key, val):
        try:
            self.TAGS_CACHE[key] = val
            self.save_tags_cache()
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)
            self.TAGS_CACHE[key] = val

    def _lsp_document_symbols(self, uri):
        abs_path = self._abs_from_uri(uri)
        mtime = self.get_mtime(abs_path) if abs_path else None
        mem_key = ("lsp", "docsyms", abs_path, mtime)
        if mem_key in self.LSP_MEM_CACHE:
            return self.LSP_MEM_CACHE[mem_key]
        disk_key = ("lsp", "docsyms", abs_path)
        if abs_path and mtime is not None:
            v = self._cache_get(disk_key)
            if isinstance(v, dict) and v.get("mtime") == mtime:
                self.LSP_MEM_CACHE[mem_key] = v.get("data") or []
                return v.get("data") or []
        res = self.lsp.document_symbols(uri)
        if not isinstance(res, list):
            res = []
        if abs_path and mtime is not None:
            self._cache_set(disk_key, {"mtime": mtime, "data": res})
        self.LSP_MEM_CACHE[mem_key] = res
        return res

    def _lsp_definition(self, uri, position):
        abs_path = self._abs_from_uri(uri)
        mtime = self.get_mtime(abs_path) if abs_path else None
        line = int(position.get("line", 0)) if isinstance(position, dict) else 0
        ch = int(position.get("character", 0)) if isinstance(position, dict) else 0
        mem_key = ("lsp", "def", abs_path, line, ch, mtime)
        if mem_key in self.LSP_MEM_CACHE:
            return self.LSP_MEM_CACHE[mem_key]
        disk_key = ("lsp", "def", abs_path, line, ch)
        if abs_path and mtime is not None:
            v = self._cache_get(disk_key)
            if isinstance(v, dict) and v.get("src_mtime") == mtime:
                ok = True
                for p, mt in (v.get("targets") or {}).items():
                    cmt = self.get_mtime(p)
                    if cmt != mt:
                        ok = False
                        break
                if ok:
                    self.LSP_MEM_CACHE[mem_key] = v.get("data")
                    return v.get("data")
        res = self.lsp.definition(uri, position)
        items = res if isinstance(res, list) else [res]
        targets = {}
        for it in items or []:
            if not isinstance(it, dict):
                continue
            t_uri = it.get("uri") or it.get("targetUri")
            t_abs = self._abs_from_uri(t_uri)
            if t_abs:
                t_m = self.get_mtime(t_abs)
                if t_m is not None:
                    targets[t_abs] = t_m
        if abs_path and mtime is not None:
            self._cache_set(disk_key, {"src_mtime": mtime, "targets": targets, "data": res})
        self.LSP_MEM_CACHE[mem_key] = res
        return res

    def _lsp_references(self, uri, position):
        abs_path = self._abs_from_uri(uri)
        mtime = self.get_mtime(abs_path) if abs_path else None
        line = int(position.get("line", 0)) if isinstance(position, dict) else 0
        ch = int(position.get("character", 0)) if isinstance(position, dict) else 0
        mem_key = ("lsp", "refs", abs_path, line, ch, mtime)
        if mem_key in self.LSP_MEM_CACHE:
            return self.LSP_MEM_CACHE[mem_key]
        disk_key = ("lsp", "refs", abs_path, line, ch)
        if abs_path and mtime is not None:
            v = self._cache_get(disk_key)
            if isinstance(v, dict) and v.get("src_mtime") == mtime:
                ok = True
                for p, mt in (v.get("targets") or {}).items():
                    cmt = self.get_mtime(p)
                    if cmt != mt:
                        ok = False
                        break
                if ok:
                    self.LSP_MEM_CACHE[mem_key] = v.get("data") or []
                    return v.get("data") or []
        res = self.lsp.references(uri, position)
        lst = res if isinstance(res, list) else []
        targets = {}
        for it in lst or []:
            if not isinstance(it, dict):
                continue
            t_uri = it.get("uri") or it.get("targetUri")
            t_abs = self._abs_from_uri(t_uri)
            if t_abs:
                t_m = self.get_mtime(t_abs)
                if t_m is not None:
                    targets[t_abs] = t_m
        if abs_path and mtime is not None:
            self._cache_set(disk_key, {"src_mtime": mtime, "targets": targets, "data": lst})
        self.LSP_MEM_CACHE[mem_key] = lst
        return lst
    def _lsp_declaration(self, uri, position):
        abs_path = self._abs_from_uri(uri)
        mtime = self.get_mtime(abs_path) if abs_path else None
        line = int(position.get("line", 0)) if isinstance(position, dict) else 0
        ch = int(position.get("character", 0)) if isinstance(position, dict) else 0
        mem_key = ("lsp", "decl", abs_path, line, ch, mtime)
        if mem_key in self.LSP_MEM_CACHE:
            return self.LSP_MEM_CACHE[mem_key]
        disk_key = ("lsp", "decl", abs_path, line, ch)
        if abs_path and mtime is not None:
            v = self._cache_get(disk_key)
            if isinstance(v, dict) and v.get("src_mtime") == mtime:
                ok = True
                for p, mt in (v.get("targets") or {}).items():
                    cmt = self.get_mtime(p)
                    if cmt != mt:
                        ok = False
                        break
                if ok:
                    self.LSP_MEM_CACHE[mem_key] = v.get("data")
                    return v.get("data")
        try:
            res = self.lsp.declaration(uri, position)
        except Exception:
            res = []
        items = res if isinstance(res, list) else [res]
        targets = {}
        for it in items or []:
            if not isinstance(it, dict):
                continue
            t_uri = it.get("uri") or it.get("targetUri")
            t_abs = self._abs_from_uri(t_uri)
            if t_abs:
                t_m = self.get_mtime(t_abs)
                if t_m is not None:
                    targets[t_abs] = t_m
        if abs_path and mtime is not None:
            self._cache_set(disk_key, {"src_mtime": mtime, "targets": targets, "data": res})
        self.LSP_MEM_CACHE[mem_key] = res
        return res
    def _pos_in_range(self, pos, rng):
        try:
            s = rng.get("start", {})
            e = rng.get("end", {})
            if not (isinstance(s, dict) and isinstance(e, dict)):
                return False
            # Compare line/character lexicographically: start <= pos < end
            ps = (pos.get("line", 0), pos.get("character", 0))
            ss = (s.get("line", 0), s.get("character", 0))
            ee = (e.get("line", 0), e.get("character", 0))
            return ss <= ps < ee
        except Exception:
            return False

    def _select_enclosing_symbol(self, symbols, pos, allowed_kinds=None):
        """Pick the smallest document symbol whose range contains pos.
        Optionally restrict to allowed_kinds (set of numeric kind codes).
        """
        best = None
        best_size = None
        for sym in symbols or []:
            if not isinstance(sym, dict):
                continue
            rng = sym.get("range") or sym.get("selectionRange")
            if not isinstance(rng, dict):
                continue
            if not self._pos_in_range(pos, rng):
                continue
            k = _kind_code(sym.get("kind"))
            if allowed_kinds and k not in allowed_kinds:
                continue
            # size as (line span, char span) to prefer tighter ranges
            s = rng.get("start", {})
            e = rng.get("end", {})
            span = (e.get("line", 0) - s.get("line", 0), e.get("character", 0) - s.get("character", 0))
            if best is None or span < best_size:
                best = sym
                best_size = span
        return best

    def _ident_for_symbol(self, rel_fname, sym):
        try:
            start = sym.get("selectionRange", {}).get("start") or sym.get("range", {}).get("start", {})
            if not isinstance(start, dict):
                return None
            return build_ident_loc(rel_fname, start)
        except Exception:
            return None
    def _is_symbol_declaration(self, sym):
        try:
            rng = sym.get("range")
            sel = sym.get("selectionRange")
            if not isinstance(rng, dict) or not isinstance(sel, dict):
                return False
            if self._range_equal(rng, sel):
                return False
            s = rng.get("start", {})
            e = rng.get("end", {})
            if not isinstance(s, dict) or not isinstance(e, dict):
                return False
            ls = int(s.get("line", 0))
            le = int(e.get("line", 0))
            # Require multi-line span to avoid import/alias one-liners
            return le > ls
        except Exception:
            return False
    def _range_equal(self, a, b):
        try:
            if not (isinstance(a, dict) and isinstance(b, dict)):
                return False
            sa, ea = a.get("start", {}), a.get("end", {})
            sb, eb = b.get("start", {}), b.get("end", {})
            return (
                sa.get("line", -1) == sb.get("line", -2)
                and sa.get("character", -1) == sb.get("character", -2)
                and ea.get("line", -1) == eb.get("line", -2)
                and ea.get("character", -1) == eb.get("character", -2)
            )
        except Exception:
            return False


    def __del__(self):
        try:
            if getattr(self, "lsp", None):
                self.lsp.close()
        except Exception:
            pass

    def _load_ignore_spec(self):
        try:
            from aider.watch import load_gitignores

            return load_gitignores([Path(self.root) / ".gitignore"])
        except Exception:
            return None

    def _is_in_workspace(self, abs_path):
        try:
            if not abs_path or "://" in str(abs_path):
                return False
            root = Path(self.root).resolve()
            p = Path(abs_path).resolve()
            return p == root or root in p.parents
        except Exception:
            return False

    def _is_project_file(self, abs_path):
        try:
            if not self._is_in_workspace(abs_path):
                return False
            root = Path(self.root).resolve()
            p = Path(abs_path).resolve()
            rel = p.relative_to(root)
            if self.ignore_spec and self.ignore_spec.match_file(rel.as_posix()):
                return False
            return True
        except Exception:
            return False

    def token_count(self, text):
        len_text = len(text)
        if len_text < 200:
            return self.main_model.token_count(text)

        lines = text.splitlines(keepends=True)
        num_lines = len(lines)
        step = num_lines // 100 or 1
        lines = lines[::step]
        sample_text = "".join(lines)
        sample_tokens = self.main_model.token_count(sample_text)
        est_tokens = sample_tokens / len(sample_text) * len_text
        return est_tokens

    def get_repo_map(
        self,
        chat_files,
        other_files,
        mentioned_fnames=None,
        mentioned_idents=None,
        force_refresh=False,
    ):
        if self.max_map_tokens <= 0:
            return
        if not other_files:
            return
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        max_map_tokens = self.max_map_tokens

        # With no files in the chat, give a bigger view of the entire repo
        padding = 4096
        if max_map_tokens and self.max_context_window:
            target = min(
                int(max_map_tokens * self.map_mul_no_files),
                self.max_context_window - padding,
            )
        else:
            target = 0
        if not chat_files and self.max_context_window and target > 0:
            max_map_tokens = target

        try:
            files_listing = self.get_ranked_tags_map(
                chat_files,
                other_files,
                max_map_tokens,
                mentioned_fnames,
                mentioned_idents,
                force_refresh,
            )
        except RecursionError:
            self.io.tool_error("Disabling repo map, git repo too large?")
            self.max_map_tokens = 0
            return

        if not files_listing:
            return

        if self.verbose:
            num_tokens = self.token_count(files_listing)
            self.io.tool_output(f"Repo-map: {num_tokens / 1024:.1f} k-tokens")

        if chat_files:
            other = "other "
        else:
            other = ""

        if self.repo_content_prefix:
            repo_content = self.repo_content_prefix.format(other=other)
        else:
            repo_content = ""

        repo_content += files_listing

        return repo_content

    def get_rel_fname(self, fname):
        try:
            return os.path.relpath(fname, self.root)
        except ValueError:
            # Issue #1288: ValueError: path is on mount 'C:', start on mount 'D:'
            # Just return the full fname.
            return fname

    def tags_cache_error(self, original_error=None):
        """Handle SQLite errors by trying to recreate cache, falling back to dict if needed"""

        if self.verbose and original_error:
            self.io.tool_warning(f"Tags cache error: {str(original_error)}")

        if isinstance(getattr(self, "TAGS_CACHE", None), dict):
            return

        path = Path(self.root) / self.TAGS_CACHE_DIR

        # Try to recreate the cache
        try:
            # Delete existing cache dir
            if path.exists():
                shutil.rmtree(path)

            # Try to create new cache
            new_cache = Cache(path)

            # Test that it works
            test_key = "test"
            new_cache[test_key] = "test"
            _ = new_cache[test_key]
            del new_cache[test_key]

            # If we got here, the new cache works
            self.TAGS_CACHE = new_cache
            return

        except SQLITE_ERRORS as e:
            # If anything goes wrong, warn and fall back to dict
            self.io.tool_warning(
                f"Unable to use tags cache at {path}, falling back to memory cache"
            )
            if self.verbose:
                self.io.tool_warning(f"Cache recreation error: {str(e)}")

        self.TAGS_CACHE = dict()

    def load_tags_cache(self):
        path = Path(self.root) / self.TAGS_CACHE_DIR
        try:
            self.TAGS_CACHE = Cache(path)
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)

    def save_tags_cache(self):
        pass

    def get_mtime(self, fname):
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            self.io.tool_warning(f"File not found error: {fname}")

    def get_tags(self, fname, rel_fname):
        # Check if the file is in the cache and if the modification time has not changed
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            return []

        cache_key = fname
        try:
            val = self.TAGS_CACHE.get(cache_key)  # Issue #1308
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)
            val = self.TAGS_CACHE.get(cache_key)

        if val is not None and val.get("mtime") == file_mtime:
            try:
                return self.TAGS_CACHE[cache_key]["data"]
            except SQLITE_ERRORS as e:
                self.tags_cache_error(e)
                return self.TAGS_CACHE[cache_key]["data"]

        # miss!
        data = list(self.get_tags_raw(fname, rel_fname))

        # Update the cache
        try:
            self.TAGS_CACHE[cache_key] = {"mtime": file_mtime, "data": data}
            self.save_tags_cache()
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)
            self.TAGS_CACHE[cache_key] = {"mtime": file_mtime, "data": data}

        return data

    def get_tags_raw(self, fname, rel_fname):
        lang = filename_to_lang(fname)
        if not lang:
            return

        try:
            language = get_language(lang)
            parser = get_parser(lang)
        except Exception as err:
            print(f"Skipping file {fname}: {err}")
            return

        query_scm = get_scm_fname(lang)
        if not query_scm.exists():
            return
        query_scm = query_scm.read_text()

        code = self.io.read_text(fname)
        if not code:
            return
        tree = parser.parse(bytes(code, "utf-8"))

        # Run the tags queries
        query = language.query(query_scm)
        captures = query.captures(tree.root_node)

        saw = set()
        if USING_TSL_PACK:
            all_nodes = []
            for tag, nodes in captures.items():
                all_nodes += [(node, tag) for node in nodes]
        else:
            all_nodes = list(captures)

        for node, tag in all_nodes:
            if tag.startswith("name.definition."):
                kind = "def"
            elif tag.startswith("name.reference."):
                kind = "ref"
            else:
                continue

            saw.add(kind)

            result = Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=node.text.decode("utf-8"),
                kind=kind,
                line=node.start_point[0],
            )

            yield result

        if "ref" in saw:
            return
        if "def" not in saw:
            return

        # We saw defs, without any refs; do not synthesize refs via token lexing,
        # to avoid noisy, non-semantic references that distort ranking.
        return

    def _lsp_is_supported_file(self, fname):
        try:
            return self.lsp.is_supported_file(fname)
        except Exception:
            return False

    def _lsp_collect_defs_refs(self, fnames, progress=None):
        defines = defaultdict(set)
        references = defaultdict(list)
        definitions = defaultdict(set)
        # ident-level edges: src_ident -> dst_ident with counts
        ident_edges = Counter()
        ident_to_file = dict()

        files = [f for f in fnames if self._lsp_is_supported_file(f)]
        if not files:
            return defines, references, definitions, ident_edges, ident_to_file

        def_locs = dict()
        # cache document symbols per file we touch
        doc_symbols_cache = {}
        # seen definition locations to avoid duplicates (abs_path, line, char)
        seen_defs = set()

        for abs_fname in files:
            if progress:
                progress(f"{UPDATING_REPO_MAP_MESSAGE}: {abs_fname}")
            uri = path_to_uri(abs_fname)
            rel_fname = self.get_rel_fname(abs_fname)
            try:
                symbols = self._lsp_document_symbols(uri)
            except Exception:
                continue

            for sym in flatten_document_symbols(symbols):
                if not isinstance(sym, dict):
                    continue
                pos = sym.get("selectionRange", {}).get("start", {})
                if not (isinstance(pos, dict) and "line" in pos and "character" in pos):
                    continue
                sym_sel_range = sym.get("selectionRange") or sym.get("range")
                sym_kind = _kind_code(sym.get("kind"))
                definitional_kinds = {
                    _kind_code("function"),
                    _kind_code("method"),
                    _kind_code("constructor"),
                    _kind_code("class"),
                    _kind_code("interface"),
                    _kind_code("enum"),
                    _kind_code("struct"),
                }
                if sym_kind not in definitional_kinds:
                    continue
                # If declaration is outside project, skip (likely import alias)
                decls = []
                try:
                    decls = self._lsp_declaration(uri, pos)
                except Exception:
                    decls = []
                is_import_like = False
                if isinstance(decls, list) and decls:
                    import_like = True
                    for loc in decls:
                        if not isinstance(loc, dict):
                            continue
                        du = loc.get("uri") or loc.get("targetUri")
                        if not du:
                            continue
                        if du == uri:
                            import_like = False
                            break
                        dpath = uri_to_path(du)
                        if self._is_project_file(dpath):
                            import_like = False
                            break
                    is_import_like = import_like
                if is_import_like:
                    continue

                try:
                    defs = self._lsp_definition(uri, pos)
                except Exception:
                    defs = []
                items = defs if isinstance(defs, list) else [defs]
                # Count valid definition items returned by LSP (before filtering)
                valid_def_items = 0
                filtered = []
                for loc in items:
                    if not isinstance(loc, dict):
                        continue
                    d_uri = loc.get("uri") or loc.get("targetUri")
                    d_full_range = loc.get("range") or loc.get("targetRange")
                    d_sel_range = loc.get("targetSelectionRange") or d_full_range
                    if not d_uri or not d_full_range or not isinstance(d_full_range, dict):
                        continue
                    valid_def_items += 1
                    is_self = (d_uri == uri) and (
                        self._pos_in_range(pos, d_sel_range or d_full_range)
                    )
                    allowed_self_kinds = {
                        _kind_code("function"),
                        _kind_code("method"),
                        _kind_code("constructor"),
                        _kind_code("class"),
                        _kind_code("interface"),
                        _kind_code("enum"),
                        _kind_code("struct"),
                    }
                    if is_self and (sym_kind not in allowed_self_kinds or not self._is_symbol_declaration(sym)):
                        continue
                    d_abs = uri_to_path(d_uri)
                    if self._is_project_file(d_abs):
                        filtered.append({"uri": d_uri, "range": d_full_range, "sel": d_sel_range})
                # Only fallback if there were no valid defs returned at all by LSP
                # and the symbol is a definitional kind (avoid imports/variables)
                if not filtered and valid_def_items == 0:
                    allowed_self_kinds = {
                        _kind_code("function"),
                        _kind_code("method"),
                        _kind_code("constructor"),
                        _kind_code("class"),
                        _kind_code("interface"),
                        _kind_code("enum"),
                        _kind_code("struct"),
                    }
                    if sym_kind in allowed_self_kinds and self._is_symbol_declaration(sym):
                        fallback_range = sym.get("selectionRange") or sym.get("range")
                        if isinstance(fallback_range, dict):
                            filtered = [{"uri": uri, "range": fallback_range}]
                # If LSP returned defs but all were filtered out (outside project), skip this symbol
                if not filtered:
                    continue
                for loc in filtered:
                    d_uri = loc.get("uri")
                    d_range = loc.get("sel") or loc.get("range")
                    d_abs = uri_to_path(d_uri)
                    d_rel = self.get_rel_fname(d_abs)
                    d_start = d_range.get("start", {})
                    line = d_start.get("line", 0)
                    ident = build_ident_loc(d_rel, d_start)

                    # Validate target really is a declaration by checking enclosing symbol kind at target
                    try:
                        if d_uri == uri:
                            if d_uri not in doc_symbols_cache:
                                try:
                                    doc_symbols_cache[d_uri] = flatten_document_symbols(self._lsp_document_symbols(d_uri))
                                except Exception:
                                    doc_symbols_cache[d_uri] = []
                            target_syms = doc_symbols_cache.get(d_uri) or []
                            if target_syms:
                                allowed_def_kinds = {
                                    _kind_code("function"),
                                    _kind_code("method"),
                                    _kind_code("constructor"),
                                    _kind_code("class"),
                                    _kind_code("interface"),
                                    _kind_code("enum"),
                                    _kind_code("struct"),
                                }
                                container = self._select_enclosing_symbol(target_syms, d_start, allowed_kinds=allowed_def_kinds)
                                if container is None or not self._is_symbol_declaration(container):
                                    continue
                    except Exception:
                        pass

                    # Deduplicate identical definition locations across all symbols
                    try:
                        abs_key = (str(Path(d_abs).resolve()), int(d_start.get("line", 0)), int(d_start.get("character", 0)))
                        if abs_key in seen_defs:
                            continue
                        seen_defs.add(abs_key)
                    except Exception:
                        pass

                    defines[ident].add(d_rel)
                    definitions[(d_rel, ident)].add(
                        Tag(
                            rel_fname=d_rel,
                            fname=uri_to_path(d_uri),
                            name=sym.get("name", ident),
                            kind="def",
                            line=line,
                        )
                    )
                    if ident not in def_locs:
                        def_locs[ident] = (d_uri, d_start)
                    ident_to_file[ident] = d_rel

        refs_queried = set()
        for ident, (d_uri, d_start) in def_locs.items():
            # Ensure we only query references once per unique definition location
            try:
                d_abs = uri_to_path(d_uri)
                key = (str(Path(d_abs).resolve()), int(d_start.get("line", 0)), int(d_start.get("character", 0)))
                if key in refs_queried:
                    continue
                refs_queried.add(key)
            except Exception:
                pass
            try:
                refs = self._lsp_references(d_uri, d_start)
            except Exception:
                refs = []
            if not isinstance(refs, list):
                continue
            for ref in refs or []:
                if not isinstance(ref, dict):
                    continue
                r_uri = ref.get("uri") or ref.get("targetUri")
                if not r_uri:
                    continue
                r_abs = uri_to_path(r_uri)
                if not self._is_project_file(r_abs):
                    continue
                r_rel = self.get_rel_fname(r_abs)
                references[ident].append(r_rel)

                # Try to map this reference to an enclosing definition in r_uri
                try:
                    rr = ref.get("range") or ref.get("targetRange") or {}
                    rpos = rr.get("start", {})
                    if r_uri not in doc_symbols_cache:
                        try:
                            doc_symbols_cache[r_uri] = flatten_document_symbols(self._lsp_document_symbols(r_uri))
                        except Exception:
                            doc_symbols_cache[r_uri] = []
                    syms = doc_symbols_cache.get(r_uri) or []
                    allowed = { _kind_code("function"), _kind_code("method"), _kind_code("constructor") }
                    container = self._select_enclosing_symbol(syms, rpos, allowed_kinds=allowed)
                    if container is None:
                        container = self._select_enclosing_symbol(syms, rpos, allowed_kinds=None)
                    if container is not None:
                        src_ident = self._ident_for_symbol(r_rel, container)
                        if src_ident:
                            ident_edges[(src_ident, ident)] += 1
                            ident_to_file[src_ident] = r_rel
                except Exception:
                    pass

        return defines, references, definitions, ident_edges, ident_to_file

    def get_ranked_tags(
        self, chat_fnames, other_fnames, mentioned_fnames, mentioned_idents, progress=None
    ):
        import networkx as nx

        defines = defaultdict(set)
        references = defaultdict(list)
        definitions = defaultdict(set)

        personalization = dict()

        fnames = set(chat_fnames).union(set(other_fnames))
        chat_rel_fnames = set()

        fnames = sorted(fnames)

        # Default personalization for unspecified files is 1/num_nodes
        # https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_analysis/pagerank_alg.html#pagerank
        personalize = 100 / len(fnames)

        try:
            cache_size = len(self.TAGS_CACHE)
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)
            cache_size = len(self.TAGS_CACHE)

        if len(fnames) - cache_size > 100:
            self.io.tool_output(
                "Initial repo scan can be slow in larger repos, but only happens once."
            )
            fnames = tqdm(fnames, desc="Scanning repo")
            showing_bar = True
        else:
            showing_bar = False

        for fname in fnames:
            if self.verbose:
                self.io.tool_output(f"Processing {fname}")
            if progress and not showing_bar:
                progress(f"{UPDATING_REPO_MAP_MESSAGE}: {fname}")

            try:
                file_ok = Path(fname).is_file()
            except OSError:
                file_ok = False

            if not file_ok:
                if fname not in self.warned_files:
                    self.io.tool_warning(f"Repo-map can't include {fname}")
                    self.io.tool_output(
                        "Has it been deleted from the file system but not from git?"
                    )
                    self.warned_files.add(fname)
                continue

            rel_fname = self.get_rel_fname(fname)
            current_pers = 0.0  # Start with 0 personalization score

            if fname in chat_fnames:
                current_pers += personalize
                chat_rel_fnames.add(rel_fname)

            if rel_fname in mentioned_fnames:
                # Use max to avoid double counting if in chat_fnames and mentioned_fnames
                current_pers = max(current_pers, personalize)

            # Check path components against mentioned_idents
            path_obj = Path(rel_fname)
            path_components = set(path_obj.parts)
            basename_with_ext = path_obj.name
            basename_without_ext, _ = os.path.splitext(basename_with_ext)
            components_to_check = path_components.union({basename_with_ext, basename_without_ext})

            matched_idents = components_to_check.intersection(mentioned_idents)
            if matched_idents:
                # Add personalization *once* if any path component matches a mentioned ident
                current_pers += personalize

            if current_pers > 0:
                personalization[rel_fname] = current_pers

            if self._lsp_is_supported_file(fname):
                continue

            tags = list(self.get_tags(fname, rel_fname))
            if tags is None:
                continue

            lang = filename_to_lang(rel_fname) or Path(rel_fname).suffix.lstrip(".") or "unknown"
            for tag in tags:
                if tag.kind == "def":
                    ident_key = f"{lang}:{tag.name}"
                    defines[ident_key].add(rel_fname)
                    key = (rel_fname, ident_key)
                    definitions[key].add(tag)

                elif tag.kind == "ref":
                    ident_key = f"{lang}:{tag.name}"
                    references[ident_key].append(rel_fname)

        lsp_defines, lsp_references, lsp_definitions, lsp_ident_edges, lsp_ident_to_file = self._lsp_collect_defs_refs(
            fnames, progress=progress
        )
        if self.verbose:
            try:
                self.io.tool_output(self.lsp.format_stats())
            except Exception:
                pass
        for k, v in lsp_defines.items():
            defines[k].update(v)
        for k, v in lsp_references.items():
            references[k].extend(v)
        for k, v in lsp_definitions.items():
            definitions[k].update(v)

        ##
        # dump(defines)
        # dump(references)
        # dump(personalization)

        # disabled: do not fabricate references when none found
        if False and not references:
            references = dict((k, list(v)) for k, v in defines.items())

        idents = set(defines.keys()).intersection(set(references.keys()))

        G = nx.MultiDiGraph()

        # Add a small self-edge for every definition that has no references
        # Helps with tree-sitter 0.23.2 with ruby, where "def greet(name)"
        # isn't counted as a def AND a ref. tree-sitter 0.24.0 does.
        for ident in defines.keys():
            if ident in references:
                continue
            if "@" not in ident:
                continue
            for definer in defines[ident]:
                G.add_edge(definer, definer, weight=0.02, ident=ident)

        for ident in idents:
            if progress:
                progress(f"{UPDATING_REPO_MAP_MESSAGE}: {ident}")

            definers = defines[ident]

            mul = 1.0
            is_lsp_ident = "@" in ident
            if not is_lsp_ident:
                mul *= 0.2

            is_snake = ("_" in ident) and any(c.isalpha() for c in ident)
            is_kebab = ("-" in ident) and any(c.isalpha() for c in ident)
            is_camel = any(c.isupper() for c in ident) and any(c.islower() for c in ident)
            if ident in mentioned_idents:
                mul *= 10
            if (is_snake or is_kebab or is_camel) and len(ident) >= 8:
                mul *= 10
            if ident.startswith("_"):
                mul *= 0.1
            if len(defines[ident]) > 5:
                mul *= 0.1

            for referencer, num_refs in Counter(references[ident]).items():
                for definer in definers:
                    # Prefer to avoid strong self-loops for non-LSP inferred refs
                    if not is_lsp_ident and referencer == definer:
                        continue

                    use_mul = mul
                    if referencer in chat_rel_fnames:
                        use_mul *= 50

                    # scale down so high freq (low value) mentions don't dominate
                    num_refs = math.sqrt(num_refs)

                    G.add_edge(referencer, definer, weight=use_mul * num_refs, ident=ident)

        if not references:
            pass

        if personalization:
            pers_args = dict(personalization=personalization, dangling=personalization)
        else:
            pers_args = dict()

        try:
            ranked = nx.pagerank(G, weight="weight", **pers_args)
        except ZeroDivisionError:
            # Issue #1536
            try:
                ranked = nx.pagerank(G, weight="weight")
            except ZeroDivisionError:
                return []
        # distribute the rank from each source node, across all of its out edges
        # Track direct (from chat files) and indirect (from non-chat files) separately
        ranked_definitions_direct = defaultdict(float)
        ranked_definitions_indirect = defaultdict(float)
        for src in G.nodes:
            if progress:
                progress(f"{UPDATING_REPO_MAP_MESSAGE}: {src}")

            src_rank = ranked[src]
            total_weight = sum(data["weight"] for _src, _dst, data in G.out_edges(src, data=True))
            # dump(src, src_rank, total_weight)
            for _src, dst, data in G.out_edges(src, data=True):
                data["rank"] = src_rank * data["weight"] / total_weight
                ident = data["ident"]
                if src in chat_rel_fnames:
                    ranked_definitions_direct[(dst, ident)] += data["rank"]
                else:
                    ranked_definitions_indirect[(dst, ident)] += data["rank"]

        ranked_tags = []
        # Merge ranks with a bias toward direct usage; keep direct-first ordering (file-level)
        file_keys = set(ranked_definitions_direct.keys()) | set(
            ranked_definitions_indirect.keys()
        )
        # Compute a combined file-level score to sort; downweight indirect
        file_scores = {}
        for key in file_keys:
            d = ranked_definitions_direct.get(key, 0.0)
            ind = ranked_definitions_indirect.get(key, 0.0)
            file_scores[key] = d + 0.5 * ind

        # Build ident-level graph from LSP edges if available
        ident_scores = {}
        direct_idents = set()
        if 'lsp_ident_edges' in locals() and lsp_ident_edges:
            Gid = nx.MultiDiGraph()
            for (src_ident, dst_ident), cnt in lsp_ident_edges.items():
                w = math.sqrt(cnt)
                Gid.add_edge(src_ident, dst_ident, weight=w)
            ident_personalization = {}
            for ident, relf in lsp_ident_to_file.items():
                if relf in chat_rel_fnames:
                    ident_personalization[ident] = 1.0
            try:
                if ident_personalization:
                    pr = nx.pagerank(Gid, weight='weight', personalization=ident_personalization, dangling=ident_personalization)
                else:
                    pr = nx.pagerank(Gid, weight='weight')
            except ZeroDivisionError:
                pr = {}
            for ident, score in pr.items():
                relf = lsp_ident_to_file.get(ident)
                if relf:
                    ident_scores[(relf, ident)] = score
            for (src_ident, dst_ident), _cnt in lsp_ident_edges.items():
                if lsp_ident_to_file.get(src_ident) in chat_rel_fnames:
                    direct_idents.add(dst_ident)

        # Final unified keys and scores combining file-level and ident-level
        all_keys = set(file_scores.keys()) | set(ident_scores.keys())
        final_scores = {}
        for key in all_keys:
            final_scores[key] = 0.3 * file_scores.get(key, 0.0) + 1.0 * ident_scores.get(key, 0.0)
        sorted_keys = sorted(all_keys, key=lambda k: (final_scores[k], k), reverse=True)

        from collections import defaultdict as _dd
        line_ranks = _dd(dict)
        direct_list = []
        indirect_list = []
        for (fname, ident) in sorted_keys:
            rnk_total = final_scores.get((fname, ident), 0.0)
            if fname in chat_rel_fnames:
                continue
            def_tags = list(definitions.get((fname, ident), []))
            for t in def_tags:
                try:
                    prev = line_ranks[t.rel_fname].get(t.line, 0)
                    if rnk_total > prev:
                        line_ranks[t.rel_fname][t.line] = rnk_total
                except Exception:
                    pass
            if (ident in direct_idents) or (ranked_definitions_direct.get((fname, ident), 0.0) > 0):
                direct_list += def_tags
            else:
                indirect_list += def_tags

        # No thresholding: include all LOIs in priority order; token search will trim
        ranked_tags = direct_list + indirect_list

        # Persist for to_tree
        try:
            self.rank_by_file_line = {k: dict(v) for k, v in line_ranks.items()}
        except Exception:
            self.rank_by_file_line = {}

        rel_other_fnames_without_tags = set(self.get_rel_fname(fname) for fname in other_fnames)

        fnames_already_included = set(rt[0] for rt in ranked_tags)
        for fname in sorted(rel_other_fnames_without_tags):
            if fname not in fnames_already_included:
                ranked_tags.append((fname,))


        top_rank = sorted([(rank, node) for (node, rank) in ranked.items()], reverse=True)
        already = {rt[0] for rt in ranked_tags}
        for rank, fname in top_rank:
            if fname not in already:
                ranked_tags.append((fname,))

        return ranked_tags

    def get_ranked_tags_map(
        self,
        chat_fnames,
        other_fnames=None,
        max_map_tokens=None,
        mentioned_fnames=None,
        mentioned_idents=None,
        force_refresh=False,
    ):
        # Create a cache key
        cache_key = [
            tuple(sorted(chat_fnames)) if chat_fnames else None,
            tuple(sorted(other_fnames)) if other_fnames else None,
            max_map_tokens,
        ]

        if self.refresh == "auto":
            cache_key += [
                tuple(sorted(mentioned_fnames)) if mentioned_fnames else None,
                tuple(sorted(mentioned_idents)) if mentioned_idents else None,
            ]
        cache_key = tuple(cache_key)

        use_cache = False
        if not force_refresh:
            if self.refresh == "manual" and self.last_map:
                return self.last_map

            if self.refresh == "always":
                use_cache = False
            elif self.refresh == "files":
                use_cache = True
            elif self.refresh == "auto":
                use_cache = self.map_processing_time > 1.0

            # Check if the result is in the cache
            if use_cache and cache_key in self.map_cache:
                return self.map_cache[cache_key]

        # If not in cache or force_refresh is True, generate the map
        start_time = time.time()
        result = self.get_ranked_tags_map_uncached(
            chat_fnames, other_fnames, max_map_tokens, mentioned_fnames, mentioned_idents
        )
        end_time = time.time()
        self.map_processing_time = end_time - start_time

        # Store the result in the cache
        self.map_cache[cache_key] = result
        self.last_map = result

        return result

    def get_ranked_tags_map_uncached(
        self,
        chat_fnames,
        other_fnames=None,
        max_map_tokens=None,
        mentioned_fnames=None,
        mentioned_idents=None,
    ):
        if not other_fnames:
            other_fnames = list()
        if not max_map_tokens:
            max_map_tokens = self.max_map_tokens
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        spin = Spinner(UPDATING_REPO_MAP_MESSAGE)

        ranked_tags = self.get_ranked_tags(
            chat_fnames,
            other_fnames,
            mentioned_fnames,
            mentioned_idents,
            progress=spin.step,
        )

        other_rel_fnames = sorted(set(self.get_rel_fname(fname) for fname in other_fnames))
        special_fnames = filter_important_files(other_rel_fnames)
        ranked_tags_fnames = set(tag[0] for tag in ranked_tags)
        special_fnames = [fn for fn in special_fnames if fn not in ranked_tags_fnames]
        special_fnames = [(fn,) for fn in special_fnames]
        ranked_tags = ranked_tags + special_fnames

        spin.step()

        num_tags = len(ranked_tags)
        lower_bound = 0
        upper_bound = num_tags
        best_tree = None
        best_tree_tokens = 0

        chat_rel_fnames = set(self.get_rel_fname(fname) for fname in chat_fnames)

        self.tree_cache = dict()

        middle = min(int(max_map_tokens // 25), num_tags)
        while lower_bound <= upper_bound:
            # dump(lower_bound, middle, upper_bound)

            if middle > 1500:
                show_tokens = f"{middle / 1000.0:.1f}K"
            else:
                show_tokens = str(middle)
            spin.step(f"{UPDATING_REPO_MAP_MESSAGE}: {show_tokens} tokens")

            tree = self.to_tree(ranked_tags[:middle], chat_rel_fnames)
            num_tokens = self.token_count(tree)

            pct_err = abs(num_tokens - max_map_tokens) / max_map_tokens
            ok_err = 0.15
            if (num_tokens <= max_map_tokens and num_tokens > best_tree_tokens) or pct_err < ok_err:
                best_tree = tree
                best_tree_tokens = num_tokens

                if pct_err < ok_err:
                    break

            if num_tokens < max_map_tokens:
                lower_bound = middle + 1
            else:
                upper_bound = middle - 1

            middle = int((lower_bound + upper_bound) // 2)

        spin.end()
        return best_tree

    tree_cache = dict()

    def render_tree(self, abs_fname, rel_fname, lois):
        mtime = self.get_mtime(abs_fname)
        key = (rel_fname, tuple(sorted(lois)), mtime)

        if key in self.tree_cache:
            return self.tree_cache[key]

        if (
            rel_fname not in self.tree_context_cache
            or self.tree_context_cache[rel_fname]["mtime"] != mtime
        ):
            code = self.io.read_text(abs_fname) or ""
            if not code.endswith("\n"):
                code += "\n"

            kwargs = dict(
                color=False,
                line_number=False,
                child_context=False,
                last_line=False,
                margin=0,
                mark_lois=False,
                loi_pad=0,
                show_top_of_file_parent_scope=False,
            )
            try:
                if self._lsp_is_supported_file(abs_fname):
                    kwargs["header_max"] = 0
            except Exception:
                pass

            context = TreeContext(
                rel_fname,
                code,
                **kwargs,
            )
            self.tree_context_cache[rel_fname] = {"context": context, "mtime": mtime}

        context = self.tree_context_cache[rel_fname]["context"]
        context.lines_of_interest = set()
        context.add_lines_of_interest(lois)
        context.add_context()
        res = context.format()
        self.tree_cache[key] = res
        return res

    def to_tree(self, tags, chat_rel_fnames):
        if not tags:
            return ""

        grouped = OrderedDict()
        abs_map = {}
        for tag in tags:
            rel = tag[0]
            if rel in chat_rel_fnames:
                continue
            if rel not in grouped:
                grouped[rel] = []
            if isinstance(tag, Tag):
                grouped[rel].append(tag.line)
                abs_map[rel] = tag.fname
            else:
                grouped.setdefault(rel, [])

        output = ""
        first = True
        for rel, lois in grouped.items():
            # Sort LOIs by importance descending, tie-break by line number
            try:
                ranks = self.rank_by_file_line.get(rel, {})
                lois = sorted(set(lois), key=lambda ln: (ranks.get(ln, 0), -ln), reverse=True)
            except Exception:
                lois = sorted(set(lois))
            if not first:
                output += "\n"
            first = False
            abs_fname = abs_map.get(rel, os.path.join(self.root, rel))
            if lois:
                output += rel + ":\n"
                output += self.render_tree(abs_fname, rel, lois)
            else:
                output += "\n" + rel + "\n"

        output = "\n".join([line[:100] for line in output.splitlines()]) + "\n"
        return output


def find_src_files(directory):
    if not os.path.isdir(directory):
        return [directory]

    src_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            src_files.append(os.path.join(root, file))
    return src_files


def get_random_color():
    hue = random.random()
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 0.75)]
    res = f"#{r:02x}{g:02x}{b:02x}"
    return res


def get_scm_fname(lang):
    # Load the tags queries
    if USING_TSL_PACK:
        subdir = "tree-sitter-language-pack"
        try:
            path = resources.files(__package__).joinpath(
                "queries",
                subdir,
                f"{lang}-tags.scm",
            )
            if path.exists():
                return path
        except KeyError:
            pass

    # Fall back to tree-sitter-languages
    subdir = "tree-sitter-languages"
    try:
        return resources.files(__package__).joinpath(
            "queries",
            subdir,
            f"{lang}-tags.scm",
        )
    except KeyError:
        return


def get_supported_languages_md():
    from grep_ast.parsers import PARSERS

    res = """
| Language | File extension | Repo map | Linter |
|:--------:|:--------------:|:--------:|:------:|
"""
    data = sorted((lang, ex) for ex, lang in PARSERS.items())

    for lang, ext in data:
        fn = get_scm_fname(lang)
        repo_map = "✓" if Path(fn).exists() else ""
        linter_support = "✓"
        res += f"| {lang:20} | {ext:20} | {repo_map:^8} | {linter_support:^6} |\n"

    res += "\n"

    return res



if __name__ == "__main__":
    fnames = sys.argv[1:]

    chat_fnames = []
    other_fnames = []
    for fname in sys.argv[1:]:
        if Path(fname).is_dir():
            chat_fnames += find_src_files(fname)
        else:
            chat_fnames.append(fname)

    rm = RepoMap(root=".")
    repo_map = rm.get_ranked_tags_map(chat_fnames, other_fnames)

    dump(len(repo_map))
    print(repo_map)
