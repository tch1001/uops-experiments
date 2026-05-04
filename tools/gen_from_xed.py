#!/usr/bin/env python3
"""Prototype XED-driven benchmark generator.

Reads `instructions.xml` from uops.info, filters down to a principled subset
of register-only instruction forms, and emits:

  - src/uops_generated.cpp     (C++ harness with one bench fn per iform)
  - results/uops_generated_meta.json (Zen 3 reference values for those iforms)

The benchmark pattern mirrors src/uops_latency_linux.cpp:
  - dependent chain (latency probe) using REP64
  - REP_TP unrolled chain across 4 destinations (throughput probe)
  - clock_gettime(CLOCK_THREAD_CPUTIME_ID) timing as primary
  - prints a table with med ns/op
"""

from __future__ import annotations

import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = os.path.join(REPO, "instructions.xml")
OUT_CPP = os.path.join(REPO, "src", "uops_generated.cpp")
OUT_META = os.path.join(REPO, "results", "uops_generated_meta.json")

ARCH = "ZEN3"

# Whitelist of extensions we touch.
ALLOWED_EXTENSIONS = {
    "BASE",
    "SSE2",
    "AVX",
    "AVX2",
    "FMA",
    "BMI1",
    "BMI2",
    "LZCNT",
    "POPCNT",
}

# Operand-token -> (class, AT&T-syntax register name).
# We pick a few fixed registers per class.
# rdx is used as the "third" GPR to keep CL/AX out of the picture.
GPR64_DEST = "%rax"
GPR64_SRC = "%rbx"
GPR64_SRC2 = "%rcx"   # only as a *source* — not for shifts where CL is special
XMM_DEST = "%xmm0"
XMM_SRC = "%xmm1"
XMM_SRC2 = "%xmm2"
YMM_DEST = "%ymm0"
YMM_SRC = "%ymm1"
YMM_SRC2 = "%ymm2"

# Throughput chain: round-robin 4 dests so deps don't serialize.
GPR64_TP_DESTS = ["%rax", "%rbx", "%rcx", "%rdx"]   # we'll redefine to not collide w sources
GPR64_TP_DESTS = ["%r8", "%r9", "%r10", "%r11"]
XMM_TP_DESTS = ["%xmm0", "%xmm1", "%xmm2", "%xmm3"]
YMM_TP_DESTS = ["%ymm0", "%ymm1", "%ymm2", "%ymm3"]
# For TP: src register is one separate from any dest.
GPR64_TP_SRC = "%r12"
XMM_TP_SRC = "%xmm5"
YMM_TP_SRC = "%ymm5"


@dataclass
class Insn:
    iform: str          # XED iform, e.g. ADD_GPRv_GPRv_01
    asm: str            # mnemonic, e.g. ADD
    extension: str
    iclass: str
    string: str         # e.g. "ADD_01 (R64, R64)"
    url_stem: str       # e.g. ADD_01_R64_R64 (the unique key)
    operand_types: list[str] = field(default_factory=list)
    # zen3 measurement
    zen3_uops: Optional[float] = None
    zen3_tp_unrolled: Optional[float] = None
    zen3_tp_loop: Optional[float] = None
    zen3_tp_ports: Optional[float] = None
    zen3_ports: str = ""
    zen3_lat_max: Optional[float] = None  # max over (start,target) pairs


# ---------------------------------------------------------------------------
# XML streaming parse
# ---------------------------------------------------------------------------

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def num(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    m = NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def latency_cycles(attrs: dict) -> Optional[float]:
    if "cycles" in attrs:
        return num(attrs["cycles"])
    lo = num(attrs.get("min_cycles"))
    hi = num(attrs.get("max_cycles"))
    if lo is not None and hi is not None:
        return (lo + hi) / 2.0
    if lo is not None:
        return lo
    if hi is not None:
        return hi
    return None


def parse_string_operands(s: str) -> list[str]:
    """Parse the human-readable "string" attr into operand tokens.

    e.g. "ADD_01 (R64, R64)" -> ["R64", "R64"]
         "VADDPD (YMM, YMM, YMM)" -> ["YMM", "YMM", "YMM"]
         "ADD (R64, I32)" -> ["R64", "I32"]
         "SHL (R64, 1)" -> ["R64", "1"]
         "SHL (R64, CL)" -> ["R64", "CL"]
         "SHL (R64, I8)" -> ["R64", "I8"]
    """
    m = re.search(r"\(([^)]*)\)", s)
    if not m:
        return []
    return [t.strip() for t in m.group(1).split(",")]


def iter_instructions(path: str):
    """Yield (instr_attrs, [(arch_name, [meas_attrs+latencies])...]) tuples."""
    # Stream parse to keep memory reasonable
    context = ET.iterparse(path, events=("start", "end"))
    cur_instr = None
    cur_arch = None
    cur_meas = None
    archs: list = []
    for event, elem in context:
        if event == "start":
            tag = elem.tag
            if tag == "instruction":
                cur_instr = dict(elem.attrib)
                archs = []
                cur_arch = None
                cur_meas = None
            elif tag == "architecture" and cur_instr is not None:
                cur_arch = elem.attrib.get("name")
            elif tag == "measurement" and cur_arch is not None:
                cur_meas = (cur_arch, dict(elem.attrib), [])
        elif event == "end":
            tag = elem.tag
            if tag == "latency" and cur_meas is not None:
                cur_meas[2].append(dict(elem.attrib))
            elif tag == "measurement" and cur_meas is not None:
                archs.append(cur_meas)
                cur_meas = None
            elif tag == "architecture":
                cur_arch = None
            elif tag == "instruction" and cur_instr is not None:
                yield cur_instr, archs, elem
                # free memory
                elem.clear()
                cur_instr = None
                archs = []


def url_stem(instr: dict) -> Optional[str]:
    url = instr.get("url", "")
    m = re.search(r"html-instr/([^.]+)\.html", url)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Filter rules
# ---------------------------------------------------------------------------

# Mnemonic families we want, mapped by iclass.
GPR_BINARY = {"ADD", "SUB", "AND", "OR", "XOR", "CMP", "TEST"}
GPR_SHIFT = {"SHL", "SHR", "SAR", "ROL", "ROR"}
BMI1_RR = {"ANDN"}
BMI1_OP = {"BLSI", "BLSR", "BLSMSK"}  # 1 source, 1 dest (RR forms)
BMI2_3OP = {"BEXTR", "SHLX", "SHRX", "SARX", "RORX", "PEXT", "PDEP", "BZHI"}
BIT_COUNT = {"POPCNT", "LZCNT", "TZCNT", "BSR", "BSF"}
SSE_SCALAR_FP = {"ADDSD", "SUBSD", "MULSD", "DIVSD", "MAXSD", "MINSD", "SQRTSD",
                 "ADDSS", "SUBSS", "MULSS", "DIVSS", "MAXSS", "MINSS", "SQRTSS"}
SSE_PACKED_FP = {"ADDPD", "SUBPD", "MULPD", "DIVPD", "MAXPD", "MINPD", "SQRTPD",
                 "ADDPS", "SUBPS", "MULPS", "DIVPS", "MAXPS", "MINPS", "SQRTPS"}
AVX_SCALAR_FP = {"VADDSD", "VSUBSD", "VMULSD", "VDIVSD", "VMAXSD", "VMINSD", "VSQRTSD",
                 "VADDSS", "VSUBSS", "VMULSS", "VDIVSS", "VMAXSS", "VMINSS", "VSQRTSS"}
AVX_PACKED_FP = {"VADDPD", "VSUBPD", "VMULPD", "VDIVPD", "VMAXPD", "VMINPD", "VSQRTPD",
                 "VADDPS", "VSUBPS", "VMULPS", "VDIVPS", "VMAXPS", "VMINPS", "VSQRTPS"}
FMA_FAMILY_PREFIXES = ("VFMADD132", "VFMADD213", "VFMADD231",
                       "VFMSUB132", "VFMSUB213", "VFMSUB231",
                       "VFNMADD132", "VFNMADD213", "VFNMADD231",
                       "VFNMSUB132", "VFNMSUB213", "VFNMSUB231")
FMA_FP_TYPES = ("PD", "PS", "SD", "SS")


def is_supported_iclass(iclass: str) -> bool:
    if iclass in GPR_BINARY: return True
    if iclass in GPR_SHIFT: return True
    if iclass == "IMUL": return True
    if iclass in BMI1_RR: return True
    if iclass in BMI1_OP: return True
    if iclass in BMI2_3OP: return True
    if iclass in BIT_COUNT: return True
    if iclass in SSE_SCALAR_FP: return True
    if iclass in SSE_PACKED_FP: return True
    if iclass in AVX_SCALAR_FP: return True
    if iclass in AVX_PACKED_FP: return True
    if any(iclass.startswith(p) and iclass[len(p):] in FMA_FP_TYPES for p in FMA_FAMILY_PREFIXES):
        return True
    return False


def filter_operands(operands: list[str]) -> Optional[str]:
    """Return a category tag describing the operand pattern, or None to skip.

    Categories used by the generator:
      - "RR_GPR64"         e.g. ADD R64, R64
      - "R_I8_GPR64"       e.g. ADD R64, I8
      - "R_I32_GPR64"      e.g. ADD R64, I32
      - "RR_I8_GPR64"      e.g. IMUL R64, R64, I8
      - "RR_I32_GPR64"     e.g. IMUL R64, R64, I32
      - "RRR_GPR64"        e.g. ANDN R64, R64, R64 ; BMI2 3-op
      - "SHIFT_R_1_GPR64"
      - "SHIFT_R_I8_GPR64"
      - "RR_XMM"           SSE2 scalar/packed XMM,XMM
      - "RR_YMM"
      - "RRR_XMM"          AVX 3-op
      - "RRR_YMM"
      - None: not a clean RR form
    """
    n = len(operands)
    if n == 2:
        a, b = operands
        if a == "R64" and b == "R64":
            return "RR_GPR64"
        if a == "R64" and b == "I8":
            return "R_I8_GPR64"
        if a == "R64" and b == "I32":
            return "R_I32_GPR64"
        if a == "R64" and b == "1":
            return "SHIFT_R_1_GPR64"
        if a == "XMM" and b == "XMM":
            return "RR_XMM"
        if a == "YMM" and b == "YMM":
            return "RR_YMM"
    elif n == 3:
        a, b, c = operands
        if a == "R64" and b == "R64" and c == "R64":
            return "RRR_GPR64"
        if a == "R64" and b == "R64" and c == "I8":
            return "RR_I8_GPR64"
        if a == "R64" and b == "R64" and c == "I32":
            return "RR_I32_GPR64"
        if a == "XMM" and b == "XMM" and c == "XMM":
            return "RRR_XMM"
        if a == "YMM" and b == "YMM" and c == "YMM":
            return "RRR_YMM"
    return None


# ---------------------------------------------------------------------------
# Codegen
# ---------------------------------------------------------------------------

@dataclass
class Bench:
    iform_key: str    # url_stem, e.g. ADD_01_R64_R64
    cpp_ident: str    # safe C++ identifier
    display: str      # e.g. "ADD_01 (R64, R64)"
    category: str     # operand pattern tag
    mnemonic: str     # AT&T mnemonic e.g. addq, vaddpd
    operands_att: str # AT&T operand string for the instruction (latency form)
    tp_blocks: list[str] = field(default_factory=list)  # AT&T lines for throughput
    tp_destinations: list[str] = field(default_factory=list)
    reg_class: str = "GPR"  # GPR / XMM / YMM
    init_kind: str = "u64"  # u64 / f64 / f32 / xmm / ymm


def att_mnemonic(asm: str, category: str) -> str:
    """Map XED asm + category -> AT&T mnemonic.

    For 64-bit GPR forms, append 'q'. For SSE/AVX scalar/packed, mnemonic
    matches case-insensitively without size suffix.
    """
    asm = asm.lower()
    if category in ("RR_GPR64", "R_I8_GPR64", "R_I32_GPR64", "RR_I8_GPR64",
                    "RR_I32_GPR64", "RRR_GPR64", "SHIFT_R_1_GPR64", "SHIFT_R_I8_GPR64"):
        # Add 'q' suffix
        return asm + "q"
    return asm


def make_cpp_ident(s: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]", "_", s)
    if not out or not out[0].isalpha():
        out = "b_" + out
    return out


def emit_bench(insn: Insn, category: str) -> Optional[Bench]:
    """Produce inline-asm lines for a benchmark, or None if we skip it."""
    asm = insn.asm
    iform = insn.iform
    # Filter out fancy {evex}, {load}, prefixes — those signal APX/EVEX which we skip.
    if asm.startswith("{") or "EVEX" in iform or "_APX" in iform:
        return None

    # The user-facing url_stem also tells us we want the basic legacy form.
    if not insn.url_stem:
        return None

    mnem = att_mnemonic(asm, category)
    cpp_ident = make_cpp_ident(insn.url_stem)
    display = insn.string

    if category == "RR_GPR64":
        # latency: dst,src ;  src=rbx, dst=rax  (dep: rax -> rax via reading)
        # AT&T:  addq %src, %dst   (dst = dst + src)
        # For CMP/TEST we have to handle non-dst forms (they only set flags,
        # so don't break the chain — skip those for now to keep scope small)
        if asm in ("CMP", "TEST"):
            # Latency chain: hard to model dst-dest because instruction
            # has no destination register. Skip latency, add throughput only?
            # For prototype: skip CMP/TEST for now.
            return None
        # Several "RR" forms are really 1-source-1-dest (the destination
        # register is purely written, not read). For those, we MUST point
        # the source at the destination so the dep chain is real.
        ONE_SOURCE = {
            "POPCNT", "LZCNT", "TZCNT",      # write dst from src; no false-dep on Zen
            "BLSI", "BLSR", "BLSMSK",        # BMI1 1-source primitives
            "BSF", "BSR",                    # write dst from src (with quirky src=0 case)
        }
        if asm in ONE_SOURCE:
            operands_att = f"{GPR64_DEST}, {GPR64_DEST}"
        else:
            operands_att = f"{GPR64_SRC}, {GPR64_DEST}"
        return Bench(insn.url_stem, cpp_ident, display, category, mnem, operands_att,
                    reg_class="GPR", init_kind="u64")

    if category == "R_I8_GPR64":
        # ADD/SUB/AND/OR/XOR/CMP/TEST r64, imm8 (sign-extended)
        if asm in ("CMP", "TEST"):
            return None
        # Use a small constant to keep value bounded for AND/OR
        imm = 7
        if asm in ("AND",):
            imm = -1   # AT&T: $-1
        operands_att = f"${imm}, {GPR64_DEST}"
        return Bench(insn.url_stem, cpp_ident, display, category, mnem, operands_att,
                    reg_class="GPR", init_kind="u64")

    if category == "R_I32_GPR64":
        if asm in ("CMP", "TEST"):
            return None
        imm = 0x12345
        operands_att = f"${imm}, {GPR64_DEST}"
        return Bench(insn.url_stem, cpp_ident, display, category, mnem, operands_att,
                    reg_class="GPR", init_kind="u64")

    if category in ("SHIFT_R_1_GPR64", "R_I8_GPR64"):
        # Already handled
        pass

    if category == "SHIFT_R_1_GPR64":
        # SHL r64, 1 — encoded with shlq $1, %rax
        operands_att = f"$1, {GPR64_DEST}"
        return Bench(insn.url_stem, cpp_ident, display, category, mnem, operands_att,
                    reg_class="GPR", init_kind="u64")

    if category == "RR_I8_GPR64":
        # IMUL r64, r64, imm8
        if asm != "IMUL":
            return None
        operands_att = f"$3, {GPR64_DEST}, {GPR64_DEST}"
        return Bench(insn.url_stem, cpp_ident, display, category, mnem, operands_att,
                    reg_class="GPR", init_kind="u64")

    if category == "RR_I32_GPR64":
        if asm != "IMUL":
            return None
        operands_att = f"$0x12345, {GPR64_DEST}, {GPR64_DEST}"
        return Bench(insn.url_stem, cpp_ident, display, category, mnem, operands_att,
                    reg_class="GPR", init_kind="u64")

    if category == "RRR_GPR64":
        # BMI1/BMI2 3-op VEX forms — dst is purely a write target.
        # Intel: OP dst, srcA, srcB     AT&T: OP srcB, srcA, dst
        # To form a latency chain we need dst to feed back as one of the
        # *read* operands. The right choice depends on the instruction:
        #  - ANDN dst, srcA, srcB = dst = ~srcA & srcB
        #      latency through srcA OR srcB. We chain via srcA (the closer src).
        #  - BEXTR dst, src, control  (control is src2)
        #      Intel: BEXTR r64, r/m64, r64
        #      AT&T:  bextrq <control>, <src>, <dst>
        #      We chain via the src (middle in AT&T).
        #  - SHLX/SHRX/SARX dst, r/m64, r64 (count)
        #      Intel: SHLX r64, r/m64, r64
        #      AT&T:  shlxq <count>, <src>, <dst>
        #      We chain via src.
        #  - PEXT/PDEP dst, srcA, mask
        #      Intel: PEXT r64, r64, r/m64
        #      AT&T:  pextq <mask>, <srcA>, <dst>
        #      We chain via srcA.
        #  - BZHI dst, r/m64, r64 (index)
        #      AT&T:  bzhiq <index>, <src>, <dst>
        #      We chain via src.
        # In every case: the *middle* AT&T operand is the one we want to
        # alias with dst. The 3rd-Intel/1st-AT&T operand stays as a fixed
        # input to provide a real dependency on something but not the dst.
        if asm == "RORX":
            return None  # encoded via RR_I8_GPR64 form
        # AT&T:  OP <fixed_input>, <chain_src=dst>, <dst>
        operands_att = f"{GPR64_SRC2}, {GPR64_DEST}, {GPR64_DEST}"
        return Bench(insn.url_stem, cpp_ident, display, category, mnem, operands_att,
                    reg_class="GPR", init_kind="u64")

    if category == "RR_XMM":
        # SSE2 scalar/packed: OP xmm_dst, xmm_src -> AT&T: op %src, %dst
        # For 2-op forms with a single source (SQRT*), dst is purely written,
        # so we must alias src to dst to keep the latency chain real.
        SSE_ONE_SRC = {"SQRTSD", "SQRTSS", "SQRTPD", "SQRTPS",
                       "VSQRTPD", "VSQRTPS"}  # 2-op AVX VSQRT* forms exist too
        if asm in SSE_ONE_SRC:
            operands_att = f"{XMM_DEST}, {XMM_DEST}"
        else:
            operands_att = f"{XMM_SRC}, {XMM_DEST}"
        kind = "f64" if "SD" in asm or "PD" in asm else "f32"
        return Bench(insn.url_stem, cpp_ident, display, category, mnem, operands_att,
                    reg_class="XMM", init_kind=kind)

    if category == "RRR_XMM":
        # AVX / FMA: VOP xmm0, xmm1, xmm2 (Intel) -> AT&T: vop %xmm2, %xmm1, %xmm0
        # For latency: want xmm0 to depend on xmm0 -> use xmm0 in place of xmm2 input.
        if asm.startswith("VFM") or asm.startswith("VFNM"):
            # FMA: dst = dst*src1 + src2 (for 132); we just want a chain.
            # Make dst appear as the accumulator: vfmaddXXX %xmm2, %xmm1, %xmm0
            operands_att = f"{XMM_SRC2}, {XMM_SRC}, {XMM_DEST}"
        else:
            # VADDPD %xmm0, %xmm1, %xmm0 -> dst depends on dst (3rd source = dst)
            operands_att = f"{XMM_DEST}, {XMM_SRC}, {XMM_DEST}"
        kind = "f64" if ("SD" in asm or "PD" in asm) else "f32"
        return Bench(insn.url_stem, cpp_ident, display, category, mnem, operands_att,
                    reg_class="XMM", init_kind=kind)

    if category == "RR_YMM":
        if asm in ("VSQRTPD", "VSQRTPS"):
            operands_att = f"{YMM_DEST}, {YMM_DEST}"
        else:
            operands_att = f"{YMM_SRC}, {YMM_DEST}"
        kind = "f64" if "PD" in asm else "f32"
        return Bench(insn.url_stem, cpp_ident, display, category, mnem, operands_att,
                    reg_class="YMM", init_kind=kind)

    if category == "RRR_YMM":
        if asm.startswith("VFM") or asm.startswith("VFNM"):
            operands_att = f"{YMM_SRC2}, {YMM_SRC}, {YMM_DEST}"
        else:
            operands_att = f"{YMM_DEST}, {YMM_SRC}, {YMM_DEST}"
        kind = "f64" if "PD" in asm else "f32"
        return Bench(insn.url_stem, cpp_ident, display, category, mnem, operands_att,
                    reg_class="YMM", init_kind=kind)

    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def collect_instructions() -> list[Insn]:
    out: list[Insn] = []
    seen_keys: set[str] = set()
    for instr_attrs, archs, elem in iter_instructions(XML_PATH):
        ext = instr_attrs.get("extension", "")
        if ext not in ALLOWED_EXTENSIONS:
            continue
        iclass = instr_attrs.get("iclass", "")
        if not is_supported_iclass(iclass):
            continue
        url = instr_attrs.get("url", "")
        if not url:
            continue
        stem = url_stem(instr_attrs)
        if not stem or stem in seen_keys:
            continue
        s = instr_attrs.get("string", "")
        operands = parse_string_operands(s)
        category = filter_operands(operands)
        if category is None:
            continue

        insn = Insn(
            iform=instr_attrs.get("iform", ""),
            asm=instr_attrs.get("asm", ""),
            extension=ext,
            iclass=iclass,
            string=s,
            url_stem=stem,
            operand_types=operands,
        )

        # Find ZEN3 measurement
        for arch_name, meas_attrs, latencies in archs:
            if arch_name != ARCH:
                continue
            insn.zen3_uops = num(meas_attrs.get("uops"))
            insn.zen3_tp_unrolled = num(meas_attrs.get("TP_unrolled"))
            insn.zen3_tp_loop = num(meas_attrs.get("TP_loop"))
            insn.zen3_tp_ports = num(meas_attrs.get("TP_ports"))
            insn.zen3_ports = meas_attrs.get("ports", "")
            cycles_seen = []
            for lat in latencies:
                c = latency_cycles(lat)
                if c is not None:
                    cycles_seen.append(c)
            if cycles_seen:
                insn.zen3_lat_max = max(cycles_seen)
            break

        seen_keys.add(stem)
        out.append(insn)
    return out


def main():
    if not os.path.exists(XML_PATH):
        sys.exit(f"missing {XML_PATH}")

    insns = collect_instructions()
    print(f"collected {len(insns)} candidate instructions after extension/iclass/operand filters")

    # Build benches
    benches: list[Bench] = []
    insn_by_key: dict[str, Insn] = {}
    for insn in insns:
        category = filter_operands(insn.operand_types)
        if category is None:
            continue
        b = emit_bench(insn, category)
        if b is None:
            continue
        if b.iform_key in insn_by_key:
            continue
        benches.append(b)
        insn_by_key[b.iform_key] = insn

    # Optional excludes — known-bad iforms we found from earlier compile attempts.
    # We populate this iteratively if assembler rejects something.
    EXCLUDES = set()
    benches = [b for b in benches if b.iform_key not in EXCLUDES]

    print(f"emitting {len(benches)} benchmarks")
    families: dict[str, int] = {}
    for b in benches:
        fam = insn_by_key[b.iform_key].iclass
        families[fam] = families.get(fam, 0) + 1
    for f in sorted(families):
        print(f"  {f}: {families[f]}")

    # Write metadata
    meta = OrderedDict()
    for b in benches:
        ins = insn_by_key[b.iform_key]
        meta[b.iform_key] = {
            "iform_xed": ins.iform,
            "asm": ins.asm,
            "iclass": ins.iclass,
            "extension": ins.extension,
            "string": ins.string,
            "operand_types": ins.operand_types,
            "category": b.category,
            "zen3": {
                "uops": ins.zen3_uops,
                "TP_unrolled": ins.zen3_tp_unrolled,
                "TP_loop": ins.zen3_tp_loop,
                "TP_ports": ins.zen3_tp_ports,
                "ports": ins.zen3_ports,
                "latency_max": ins.zen3_lat_max,
            },
        }
    os.makedirs(os.path.dirname(OUT_META), exist_ok=True)
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {OUT_META}")

    # Write C++
    write_cpp(benches, insn_by_key)
    print(f"wrote {OUT_CPP}")


# ---------------------------------------------------------------------------
# C++ harness emission
# ---------------------------------------------------------------------------

CPP_PREAMBLE = r"""// AUTO-GENERATED by tools/gen_from_xed.py — DO NOT HAND-EDIT.
// Microbenchmarks for a subset of x86 instructions, modeled on
// src/uops_latency_linux.cpp's REP64 dependent-chain pattern.

#define _GNU_SOURCE

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

#include <x86intrin.h>

#define REP2(x)  x x
#define REP4(x)  REP2(x) REP2(x)
#define REP8(x)  REP4(x) REP4(x)
#define REP16(x) REP8(x) REP8(x)
#define REP32(x) REP16(x) REP16(x)
#define REP64(x) REP32(x) REP32(x)
static constexpr int OPS_PER_ITER = 64;

static volatile uint64_t sink_u64 = 0x9e3779b97f4a7c15ull;
static volatile double   sink_f64 = 1.0000001;
static volatile float    sink_f32 = 1.0000001f;

using BenchFn = uint64_t (*)(int);

static uint64_t read_thread_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
           static_cast<uint64_t>(ts.tv_nsec);
}

static uint64_t read_tsc_begin() { _mm_lfence(); return __rdtsc(); }
static uint64_t read_tsc_end()   { unsigned aux = 0; uint64_t t = __rdtscp(&aux); _mm_lfence(); return t; }

static bool pin_to_logical_cpu(unsigned cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n & 1u) ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

__attribute__((noinline)) static uint64_t bench_empty(int iters) {
    uint64_t x = sink_u64;
    for (int i = 0; i < iters; ++i) {
        asm volatile("" : [x] "+r"(x) : : "cc");
    }
    sink_u64 = x;
    return x;
}
"""


def gpr_init() -> str:
    return r"""    register uint64_t x asm("rax") = sink_u64 | 1ull;
    register uint64_t y asm("rbx") = 0x9e3779b97f4a7c15ull;
    register uint64_t z asm("rcx") = 0x123456789abcdef1ull;
    asm volatile("" : "+r"(x), "+r"(y), "+r"(z));
"""


def gpr_finish() -> str:
    return r"""    asm volatile("" : "+r"(x), "+r"(y), "+r"(z));
    sink_u64 = x ^ y ^ z;
"""


def xmm_init_f64() -> str:
    return r"""    double dx = sink_f64;
    double dy = 1.0000000001;
    double dz = 1.5;
    register double x0 asm("xmm0") = dx;
    register double x1 asm("xmm1") = dy;
    register double x2 asm("xmm2") = dz;
    asm volatile("" : "+x"(x0), "+x"(x1), "+x"(x2));
"""


def xmm_finish_f64() -> str:
    return r"""    asm volatile("" : "+x"(x0), "+x"(x1), "+x"(x2));
    sink_f64 = x0 + x1 + x2;
"""


def xmm_init_f32() -> str:
    return r"""    float fx = sink_f32;
    float fy = 1.0000000001f;
    float fz = 1.5f;
    register float x0 asm("xmm0") = fx;
    register float x1 asm("xmm1") = fy;
    register float x2 asm("xmm2") = fz;
    asm volatile("" : "+x"(x0), "+x"(x1), "+x"(x2));
"""


def xmm_finish_f32() -> str:
    return r"""    asm volatile("" : "+x"(x0), "+x"(x1), "+x"(x2));
    sink_f32 = x0 + x1 + x2;
"""


# YMM versions: declare YMM regs by clobbering them and loading values
# We use vbroadcastsd from sink_f64 to populate ymm0/ymm1/ymm2.
def ymm_init_f64() -> str:
    return r"""    double seed_d = sink_f64;
    double seed_e = 1.0000000001;
    double seed_f = 1.5;
    asm volatile(
        "vbroadcastsd %[a], %%ymm0\n\t"
        "vbroadcastsd %[b], %%ymm1\n\t"
        "vbroadcastsd %[c], %%ymm2\n\t"
        : : [a] "x"(seed_d), [b] "x"(seed_e), [c] "x"(seed_f)
        : "ymm0", "ymm1", "ymm2");
"""


def ymm_finish_f64() -> str:
    return r"""    double out_d = 0.0;
    asm volatile(
        "vmovsd %%xmm0, %[o]\n\t"
        : [o] "=m"(out_d) : : "memory");
    sink_f64 = out_d;
"""


def ymm_init_f32() -> str:
    return r"""    float seed_d = sink_f32;
    float seed_e = 1.0000000001f;
    float seed_f = 1.5f;
    asm volatile(
        "vbroadcastss %[a], %%ymm0\n\t"
        "vbroadcastss %[b], %%ymm1\n\t"
        "vbroadcastss %[c], %%ymm2\n\t"
        : : [a] "x"(seed_d), [b] "x"(seed_e), [c] "x"(seed_f)
        : "ymm0", "ymm1", "ymm2");
"""


def ymm_finish_f32() -> str:
    return r"""    float out_f = 0.0f;
    asm volatile(
        "vmovss %%xmm0, %[o]\n\t"
        : [o] "=m"(out_f) : : "memory");
    sink_f32 = out_f;
"""


def emit_bench_function(b: Bench) -> str:
    """Return a C++ function definition for one benchmark."""
    fn_name = f"bench_{b.cpp_ident}"
    # Inside an asm() block with operand binding, literal '%' must be doubled.
    operands_escaped = b.operands_att.replace("%", "%%")
    rep_line = f'"{b.mnemonic} {operands_escaped}\\n\\t"'
    asm_block = f"REP64({rep_line})"

    if b.reg_class == "GPR":
        # The chain register is rax; we pass rbx and rcx as live sources,
        # marking them as inputs so the compiler doesn't squash them.
        # Use clobber lists that include "cc" and the registers we touch.
        body = gpr_init()
        body += f"""    for (int i = 0; i < iters; ++i) {{
        asm volatile({asm_block}
                     : "+r"(x)
                     : "r"(y), "r"(z)
                     : "cc");
    }}
"""
        body += gpr_finish()
        body += "    return x;\n"
    elif b.reg_class == "XMM":
        if b.init_kind == "f64":
            body = xmm_init_f64()
            body += f"""    for (int i = 0; i < iters; ++i) {{
        asm volatile({asm_block}
                     : "+x"(x0)
                     : "x"(x1), "x"(x2)
                     : "cc");
    }}
"""
            body += xmm_finish_f64()
            body += "    return (uint64_t)x0;\n"
        else:
            body = xmm_init_f32()
            body += f"""    for (int i = 0; i < iters; ++i) {{
        asm volatile({asm_block}
                     : "+x"(x0)
                     : "x"(x1), "x"(x2)
                     : "cc");
    }}
"""
            body += xmm_finish_f32()
            body += "    return (uint64_t)x0;\n"
    elif b.reg_class == "YMM":
        if b.init_kind == "f64":
            body = ymm_init_f64()
            body += f"""    for (int i = 0; i < iters; ++i) {{
        asm volatile({asm_block}
                     :
                     :
                     : "ymm0", "ymm1", "ymm2", "cc");
    }}
"""
            body += ymm_finish_f64()
            body += "    return 0;\n"
        else:
            body = ymm_init_f32()
            body += f"""    for (int i = 0; i < iters; ++i) {{
        asm volatile({asm_block}
                     :
                     :
                     : "ymm0", "ymm1", "ymm2", "cc");
    }}
"""
            body += ymm_finish_f32()
            body += "    return 0;\n"
    else:
        return ""

    return (
        f"__attribute__((noinline)) static uint64_t {fn_name}(int iters) {{\n"
        f"{body}"
        f"}}\n\n"
    )


def write_cpp(benches: list[Bench], insn_by_key: dict[str, Insn]) -> None:
    parts = [CPP_PREAMBLE, "\n"]

    for b in benches:
        parts.append(emit_bench_function(b))

    # Registry
    parts.append("struct BenchEntry {\n")
    parts.append("    const char* iform;\n")
    parts.append("    const char* display;\n")
    parts.append("    const char* iclass;\n")
    parts.append("    const char* category;\n")
    parts.append("    BenchFn fn;\n")
    parts.append("};\n\n")
    parts.append("static const BenchEntry kBenchmarks[] = {\n")
    for b in benches:
        ins = insn_by_key[b.iform_key]
        # Escape strings
        def esc(s: str) -> str:
            return s.replace("\\", "\\\\").replace('"', '\\"')
        parts.append(
            f'    {{"{esc(b.iform_key)}", "{esc(b.display)}", "{esc(ins.iclass)}", '
            f'"{esc(b.category)}", &bench_{b.cpp_ident}}},\n'
        )
    parts.append("};\n")
    parts.append("static constexpr int kBenchCount = sizeof(kBenchmarks) / sizeof(kBenchmarks[0]);\n\n")

    # main()
    parts.append(r"""
struct Timing { uint64_t tsc=0, thread_ns=0; };

static Timing measure_once(BenchFn fn, int iters) {
    uint64_t tn0 = read_thread_ns();
    uint64_t t0 = read_tsc_begin();
    fn(iters);
    uint64_t t1 = read_tsc_end();
    uint64_t tn1 = read_thread_ns();
    return Timing{t1 - t0, tn1 - tn0};
}

static void run_one(const BenchEntry& b, int iters, int samples,
                    double empty_thread_med, double empty_tsc_med) {
    const double ops = static_cast<double>(iters) * OPS_PER_ITER;
    for (int i = 0; i < 4; ++i) measure_once(b.fn, iters);
    std::vector<double> ns;
    std::vector<double> tsc;
    ns.reserve(samples); tsc.reserve(samples);
    for (int i = 0; i < samples; ++i) {
        Timing t = measure_once(b.fn, iters);
        ns.push_back((static_cast<double>(t.thread_ns) - empty_thread_med) / ops);
        tsc.push_back((static_cast<double>(t.tsc) - empty_tsc_med) / ops);
    }
    double n_med = median(ns);
    double t_med = median(tsc);
    std::cout << std::left << std::setw(40) << b.iform
              << std::setw(8)  << b.iclass
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(10) << n_med
              << std::setw(10) << t_med
              << "\n";
}

int main(int argc, char** argv) {
    unsigned cpu = 0;
    int iters = 1024;
    int samples = 21;
    if (argc > 1) cpu = (unsigned)std::strtoul(argv[1], nullptr, 10);
    if (argc > 2) iters = std::max(1, std::atoi(argv[2]));
    if (argc > 3) samples = std::max(3, std::atoi(argv[3]));

    bool pinned = pin_to_logical_cpu(cpu);
    usleep(10000);

    for (int i = 0; i < 16; ++i) measure_once(bench_empty, iters);
    std::vector<double> e_ns, e_tsc;
    e_ns.reserve(samples); e_tsc.reserve(samples);
    for (int i = 0; i < samples; ++i) {
        Timing t = measure_once(bench_empty, iters);
        e_ns.push_back(static_cast<double>(t.thread_ns));
        e_tsc.push_back(static_cast<double>(t.tsc));
    }
    double e_ns_med = median(e_ns);
    double e_tsc_med = median(e_tsc);

    std::cout << "uops_generated harness — XED-driven prototype\n";
    std::cout << "logical_cpu_requested=" << cpu
              << " pinned=" << (pinned ? "yes" : "no")
              << " current_cpu=" << sched_getcpu()
              << " bench_count=" << kBenchCount
              << " samples=" << samples
              << " iters=" << iters << "\n";
    std::cout << "empty_loop_median: thread_ns=" << (uint64_t)e_ns_med
              << " tsc=" << (uint64_t)e_tsc_med << "\n\n";
    std::cout << std::left << std::setw(40) << "iform"
              << std::setw(8) << "iclass"
              << std::right << std::setw(10) << "med_ns"
              << std::setw(10) << "med_tsc" << "\n";
    std::cout << std::string(68, '-') << "\n";

    for (int i = 0; i < kBenchCount; ++i) {
        run_one(kBenchmarks[i], iters, samples, e_ns_med, e_tsc_med);
    }
    return 0;
}
""")
    with open(OUT_CPP, "w") as f:
        f.write("".join(parts))


if __name__ == "__main__":
    main()
