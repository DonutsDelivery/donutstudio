#!/usr/bin/env python3
"""Writes the Metal Path Trace kernel source as one text, embedded in a C++ file.

device_metal.metal's quoted includes are inlined, each file once (#pragma once),
recursively, relative to the including file. C++ standard headers are dropped:
the kernel's shims stand in for them. Metal's own <metal_*> headers stay.

Conditionals on the macros the kernel defines (METAL_DEFINED) or that no Metal
compile has (METAL_UNDEFINED) are decided here, and the branch Metal does not take
is left out, with any include inside it. So a host-only header behind
`#if ! defined(PATHTRACE_DEVICE_METAL)` never reaches the Metal compiler, and
cannot use up its #pragma once before a branch that does need it. Any other
conditional is kept as written.

Metal names an address space on every pointer and reference. The shared code
spells out the ones that point at uploaded memory (PT_GLOBAL, PT_TABLE_SPACE,
devices/device_annotations.h); any declarator still naming none gets `thread`,
the space of a function's own values.

The Metal compiler takes C++14 with Metal's rules, so metalize() then rewrites
what the shared C++17 says otherwise. A namespace named `device` (the address
space keyword) becomes `ptdevice`. `namespace a::b {` opens one namespace at a
time. A variable at namespace scope, or a static one in a class, that is const
or constexpr goes in the `constant` address space, where Metal keeps
program-scope data.

Usage: metal_kernel_source.py <device_metal.metal> <output.cpp> <symbol> [<depfile>]
tests/pathtrace_metal_kernel_source_contract.py imports assemble().
"""
import pathlib
import re
import sys

INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]')
METAL_DEFINED = {"PATHTRACE_DEVICE_METAL", "__CUDACC__"}
METAL_UNDEFINED = {"__HIPCC__", "__CUDA_ARCH__", "SYCL_LANGUAGE_VERSION", "__SYCL_DEVICE_ONLY__",
                   "PATHTRACE_DEVICE_ONEAPI", "_MSC_VER", "ARBIT_HAVE_OPTIX"}
CONDITION = re.compile(r"^\s*#\s*(?:if\s+(!?)\s*defined\s*\(?\s*(\w+)\s*\)?|(ifn?def)\s+(\w+))\s*(?://.*)?$")

SPACES = r"(?:device|constant|thread|threadgroup|ray_data|PT_GLOBAL|PT_TABLE_SPACE|PT_THREAD)"
# A type written against its & or * and followed by a name, a comma, a bracket or
# the line's end: the code's declarator style. `a * b` keeps its spaces.
DECLARATOR = re.compile(
    r"(?<![\w.>:])((?:const\s+)?)"
    r"((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*(?:<[^<>;(){}]*(?:<[^<>;(){}]*>[^<>;(){}]*)*>)?)"
    r"([&*]+)(?=\s+[A-Za-z_(]|\s*[,)>]|\s*$)")
NOT_TYPES = {"return", "operator", "decltype", "sizeof", "case", "else", "delete", "throw", "this"}
QUALIFIED_BEFORE = re.compile(SPACES + r"\s+(?:const\s+)?$")


def condition(line):
    """True or False for a conditional this script decides, None for any other."""
    match = CONDITION.match(line)
    if match is None:
        return None
    negated = match.group(1) == "!" or match.group(3) == "ifndef"
    name = match.group(2) or match.group(4)
    if name in METAL_DEFINED:
        return not negated
    if name in METAL_UNDEFINED:
        return negated
    return None


def qualify(line):
    """Adds `thread` to each pointer or reference declarator that names no space."""
    if line.lstrip().startswith(("#", "//")):
        return line
    out, last = [], 0
    for match in DECLARATOR.finditer(line):
        type_name = match.group(2)
        if type_name in NOT_TYPES or type_name.startswith("std::forward"):
            continue
        before = line[:match.start()]
        if QUALIFIED_BEFORE.search(before) or re.match(SPACES + r"\b(?!::)", type_name):
            continue
        out.append(line[last:match.start()])
        out.append("thread ")
        last = match.start()
    out.append(line[last:])
    return "".join(out)


def inline(path, seen, out, deps):
    path = path.resolve()
    if path in seen:
        return
    seen.append(path)
    deps.append(path)
    out.append(f"// ---- {path.name}\n")
    stack = []  # per open conditional: None when kept as written, else whether its branch is taken
    for line in path.read_text().splitlines(keepends=True):
        directive = line.strip()
        if re.match(r"#\s*if", directive):
            decided = condition(line)
            stack.append(decided)
            if decided is None and all(s is not False for s in stack[:-1]):
                out.append(line)
            continue
        if re.match(r"#\s*elif\b", directive):
            if stack and stack[-1] is not None:
                raise SystemExit(f"{path}: #elif after a Metal conditional is not supported")
            if all(s is not False for s in stack):
                out.append(line)
            continue
        if re.match(r"#\s*else\b", directive):
            if stack and stack[-1] is not None:
                stack[-1] = not stack[-1]
            elif all(s is not False for s in stack[:-1]):
                out.append(line)
            continue
        if re.match(r"#\s*endif\b", directive):
            decided = stack.pop()
            if decided is None and all(s is not False for s in stack):
                out.append(line)
            continue
        if any(s is False for s in stack):
            continue
        if re.match(r"^\s*#\s*pragma\s+once", line):
            continue
        match = INCLUDE.match(line)
        if match is None:
            out.append(qualify(line))
        elif match.group(1) == '"':
            inline(path.parent / match.group(2), seen, out, deps)
        elif match.group(2).startswith("metal_"):
            out.append(line)
    if stack:
        raise SystemExit(f"{path}: unbalanced conditional")


def blank(text):
    """The text with comments, literals and preprocessor lines turned to spaces."""
    def spaces(match):
        return re.sub(r"[^\n]", " ", match.group(0))
    pattern = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])+'|^[ \t]*#(?:[^\n]*\\\n)*[^\n]*",
                         re.S | re.M)
    return pattern.sub(spaces, text)


SPECIFIERS = re.compile(r"^((?:(?:inline|static|extern)\s+)*)(constexpr|const)\b")
NOT_VARIABLE = re.compile(r"\b(?:constant|__constant__|PT_DEVICE_CONSTANT|typedef|using|friend|return|namespace|struct"
                          r"|class|enum|union|operator|template)\b")


def statement_head(code, start):
    """Where a statement's declaration begins, past access labels and template heads."""
    while True:
        match = re.match(r"\s*(?:(?:public|private|protected)\s*:(?!:))?\s*", code[start:])
        start += match.end()
        if not code.startswith("template", start) or not re.match(r"template\s*<", code[start:]):
            return start
        depth, at = 0, code.index("<", start)
        while at < len(code):
            depth += {"<": 1, ">": -1}.get(code[at], 0)
            at += 1
            if depth == 0:
                break
        start = at


def constant_variable(code, start, end, in_class):
    """Where to write `constant` in a const or constexpr variable declared in code[start:end], or None."""
    start = statement_head(code, start)
    declaration = code[start:end].split("=", 1)[0]
    if "(" in declaration or NOT_VARIABLE.search(declaration):
        return None
    match = SPECIFIERS.match(declaration)
    if match is None or (in_class and "static" not in match.group(1)):
        return None
    return start + match.start(2)


def metalize(text):
    """The C++17 shared code as the Metal compiler takes it (see the module text)."""
    text = re.sub(r"\bdevice::", "ptdevice::", text)
    text = re.sub(r"(\bnamespace\s+(?:\w+\s*::\s*)*)device\b", r"\1ptdevice", text)
    code = blank(text)
    edits = []  # (position, characters replaced, new text)
    scopes = []  # per open brace: (kind, the closing text)
    start = 0
    for at, char in enumerate(code):
        if char not in "{};":
            continue
        kind = scopes[-1][0] if scopes else "namespace"
        if char == ";" or char == "{":
            if kind in ("namespace", "class"):
                place = constant_variable(code, start, at, kind == "class")
                if place is not None:
                    edits.append((place, 0, "constant "))
        if char == "{":
            head = code[statement_head(code, start):at]
            nested = re.match(r"\s*(?:inline\s+)?namespace\s+(\w+(?:\s*::\s*\w+)+)\s*$", head)
            if nested is not None:
                names = [name.strip() for name in nested.group(1).split("::")]
                edits.append((start + len(code[start:at]) - len(head) + nested.start(1), len(nested.group(1)),
                              " { namespace ".join(names)))
                scopes.append(("namespace", "}" * len(names)))
            elif re.match(r"\s*(?:inline\s+)?namespace\b", head) or re.match(r"\s*extern\s*$", head):
                scopes.append(("namespace", "}"))
            elif re.search(r"\b(?:struct|class|union)\b", head) and "(" not in head and not re.search(r"\benum\b", head):
                scopes.append(("class", "}"))
            else:
                scopes.append(("other", "}"))
        elif char == "}":
            if scopes:
                closing = scopes.pop()[1]
                if closing != "}":
                    edits.append((at, 1, closing))
        start = at + 1
    for position, length, new in sorted(edits, reverse=True):
        text = text[:position] + new + text[position + length:]
    return text


def assemble(source):
    """The kernel text and every file it read."""
    seen, out, deps = [], [], []
    inline(pathlib.Path(source), seen, out, deps)
    return metalize("".join(out)), deps


def main():
    source, output, symbol = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
    text, deps = assemble(source)
    data = text.encode()
    rows = ",\n".join(",".join(f"0x{b:02x}" for b in data[i:i + 32]) for i in range(0, len(data), 32))
    output.write_text(
        "#include <cstddef>\n"
        f"extern const unsigned char {symbol}[];\nextern const std::size_t {symbol}Size;\n"
        f"const unsigned char {symbol}[] = {{\n{rows},\n0x00 }};\n"
        f"const std::size_t {symbol}Size = {len(data)};\n")
    if len(sys.argv) > 4:
        dep_list = " ".join(str(p).replace(" ", "\\ ") for p in deps)
        pathlib.Path(sys.argv[4]).write_text(f"{str(output).replace(' ', chr(92) + ' ')}: {dep_list}\n")


if __name__ == "__main__":
    main()
