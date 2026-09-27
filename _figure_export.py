#!/usr/bin/env python3
"""Publication figure export helpers (figure_export.py)."""

from __future__ import annotations
import os
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

MM_PER_INCH = 25.4

# Per-journal specs: width_mm keys (single, double/full); min_dpi for line art; min_font_pt final size floor
JOURNAL_SPECS = {
    "nature": {
        "single_mm": 89.0,
        "double_mm": 183.0,
        "min_dpi_line": 600,
        "min_dpi_halftone": 300,
        "min_dpi_combo": 600,
        "min_font_pt": 5.0,
        "preferred_formats": ("pdf", "tiff", "eps"),
        "font_family": "sans-serif",
        "verified": True,
        "note": "Nature author guide: 89 mm single / 183 mm double column; "
        "panel labels 8 pt bold; body text max 7 pt, min 5 pt; Helvetica/Arial; "
        "photos 300-600 dpi; keep text as text",
    },
    "science": {
        "single_mm": 55.0,
        "double_mm": 120.0,
        "full_mm": 183.0,
        "min_dpi_line": 600,
        "min_dpi_halftone": 300,
        "min_dpi_combo": 500,
        "min_font_pt": 5.0,
        "preferred_formats": ("eps", "pdf", "ai", "tiff"),
        "font_family": "sans-serif",
        "verified": True,
        "note": "Science/AAAS three column widths (author guide): "
        "1-col 5.5 cm = 55 mm / 2-col 12 cm = 120 mm / full page 18.3 cm = 183 mm. "
        "DPI details not individually verified.",
    },
    "cell": {
        "single_mm": 85.0,
        "double_mm": 174.0,
        "min_dpi_line": 1000,
        "min_dpi_halftone": 300,
        "min_dpi_combo": 500,
        "min_font_pt": 5.0,
        "preferred_formats": ("pdf", "ai", "eps", "tiff"),
        "font_family": "sans-serif",
        "verified": False,
        "note": "Cell Press digital art guidelines; line art often 1000 dpi; not individually verified",
    },
    "plos": {
        "single_mm": 83.0,
        "onehalf_mm": 140.0,
        "double_mm": 190.0,
        "min_dpi_line": 600,
        "min_dpi_halftone": 300,
        "min_dpi_combo": 600,
        "min_font_pt": 8.0,
        "preferred_formats": ("tiff", "eps"),
        "max_height_px": 2625,
        "max_file_mb": 10.0,
        "font_family": "sans-serif",
        "verified": True,
        "note": "PLOS figure guide: TIFF/EPS only; width 789-2250 px at 300 dpi "
        "(6.68-19.05 cm); height <= 2625 px; 300-600 dpi; text Arial/Times/Symbol "
        "8-12 pt; < 10 MB",
    },
    "ieee": {
        "single_mm": 88.9,
        "double_mm": 181.0,
        "min_dpi_line": 600,
        "min_dpi_halftone": 300,
        "min_dpi_combo": 600,
        "min_font_pt": 8.0,
        "preferred_formats": ("pdf", "eps", "tiff"),
        "font_family": "sans-serif",
        "verified": False,
        "note": "IEEE two-column template widths ~3.5/7.16 in; Graphics Checker suggests >= 300 dpi",
    },
    "elsevier": {
        "single_mm": 90.0,
        "double_mm": 190.0,
        "onehalf_mm": 140.0,
        "min_dpi_line": 1000,
        "min_dpi_halftone": 300,
        "min_dpi_combo": 500,
        "min_font_pt": 7.0,
        "preferred_formats": ("pdf", "eps", "tiff"),
        "font_family": "sans-serif",
        "verified": False,
        "note": "Elsevier artwork guidelines: line art 1000 dpi, greyscale/colour 300 dpi, combination 500 dpi; not individually verified",
    },
    "mdpi": {
        "single_mm": 170.0,
        "full_mm": 170.0,
        "min_dpi_line": 1000,
        "min_dpi_halftone": 300,
        "min_dpi_combo": 1000,
        "min_font_pt": 8.0,
        "preferred_formats": ("tiff", "png", "eps"),
        "font_family": "sans-serif",
        "verified": False,
        "column_caveat": "MDPI uses a single-column layout: single/full are both the "
        "170 mm full text width, so column does not change width (unlike Nature-style "
        "journals). For narrower figures set custom_width_mm to a fraction of 170 mm.",
        "note": "MDPI single-column layout, text width ~170 mm (figures at that width "
        "or integer fractions); line art 1000 dpi, photos >= 300 dpi; TIFF/PNG/EPS; "
        "check the current author guide before submission.",
    },
}


def mm_to_inch(mm: float) -> float:
    return mm / MM_PER_INCH


def inch_to_mm(inch: float) -> float:
    return inch * MM_PER_INCH


def save_publication_figure(
    fig,
    basename,
    formats=("pdf", "png", "svg"),
    dpi=600,
    transparent=False,
    pad_inches=0.02,
    close=False,
    bbox_inches="tight",
):
    """Export in multiple formats at a given DPI. Returns the list of written paths.

    - basename may contain directories, no extension.
    - Vector formats (pdf/svg/eps) ignore dpi; kept for signature compatibility.
    - Forces pdf/ps fonttype 42 and non-outlined svg text so text stays editable.
    - bbox_inches: default "tight" trims surrounding whitespace.
      None writes exactly at fig.set_size_inches so the physical width is preserved
      (required for exact column-width submission; pad_inches has no effect then).
    """
    os.makedirs(os.path.dirname(os.path.abspath(basename)), exist_ok=True)
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    # savefig(bbox_inches=None) is not "no crop" in matplotlib - it falls back to
    # rcParams["savefig.bbox"]. If the active style sets savefig.bbox=tight, an
    # explicit None would be silently overridden, so mirror the caller's intent
    # into rcParams here to keep the explicit parameter effective.
    written = []
    prev_bbox = plt.rcParams.get("savefig.bbox")
    plt.rcParams["savefig.bbox"] = bbox_inches
    try:
        for fmt in formats:
            path = f"{basename}.{fmt}"
            save_kwargs = {
                "format": fmt,
                "dpi": dpi,
                "transparent": transparent,
                "bbox_inches": bbox_inches,
            }
            # pad_inches has no effect with bbox_inches=None; do not pass it
            if bbox_inches is not None:
                save_kwargs["pad_inches"] = pad_inches
            fig.savefig(path, **save_kwargs)
            written.append(path)
    finally:
        plt.rcParams["savefig.bbox"] = prev_bbox
    if close:
        plt.close(fig)
    return written


def _check_font_availability():
    """Check whether the preferred sans-serif font is actually available; warn if
    matplotlib silently falls back to DejaVu. Linux/CI often lacks Arial/Helvetica.
    Returns {requested, resolved, fell_back, warning}."""
    import matplotlib.font_manager as fm
    import matplotlib as mpl

    prefs = list(mpl.rcParams.get("font.sans-serif", []))
    # only check the first choice within the sans-serif family
    if not prefs:
        return {"requested": [], "resolved": None, "fell_back": False, "warning": None}
    try:
        resolved_path = fm.findfont(
            fm.FontProperties(family=prefs), fallback_to_default=True
        )
        resolved_name = fm.FontProperties(fname=resolved_path).get_name()
    except Exception:
        resolved_name = None
    # does the resolved font match any preference (case/space tolerant)
    def norm(value):
        return (value or "").lower().replace(" ", "")

    ok = any(norm(p) == norm(resolved_name) for p in prefs)
    warning = None
    if not ok:
        warning = (
            f"Font fallback: rcParams preferred {prefs[:3]} unavailable, "
            f"rendering fell back to '{resolved_name}' (typical on Linux/CI "
            f"without Arial/Helvetica); install the font or re-check in the "
            f"submission environment"
        )
    return {
        "requested": prefs[:3],
        "resolved": resolved_name,
        "fell_back": not ok,
        "warning": warning,
    }


def save_for_journal(
    fig,
    basename,
    journal="nature",
    column="single",
    height_mm=None,
    formats=None,
    dpi=None,
    custom_width_mm=None,
    **kwargs,
):
    """Set physical size from the target journal spec and export.

    - journal: a JOURNAL_SPECS key, or "custom" (with custom_width_mm).
    - column: 'single' / 'double' / 'full' / 'onehalf' (per journal).
    - height_mm: keep the current aspect ratio if not given.
    - custom_width_mm: escape hatch when journal="custom"; use a width from the
      target journal's guide, never a guess. dpi/formats default to 600 dpi and
      pdf+png; min_font is not enforced.
    Returns (written_paths, info_dict).
    """
    j = journal.lower()
    if j == "custom" or custom_width_mm is not None:
        if custom_width_mm is None:
            raise ValueError("journal='custom' requires custom_width_mm (mm)")
        width_mm = float(custom_width_mm)
        spec = {
            "min_font_pt": None,
            "verified": False,
            "note": f"custom width {width_mm} mm (caller-provided)",
            "preferred_formats": ("pdf", "png"),
            "min_dpi_line": 600,
        }
    else:
        if j not in JOURNAL_SPECS:
            raise ValueError(
                f"unknown journal '{journal}'; options: {list(JOURNAL_SPECS)} or 'custom' + custom_width_mm"
            )
        spec = JOURNAL_SPECS[j]
        key = f"{column}_mm"
        if key not in spec:
            avail = [k.replace("_mm", "") for k in spec if k.endswith("_mm")]
            raise ValueError(f"{journal} has no '{column}' width; available: {avail}")
        width_mm = spec[key]
    width_in = mm_to_inch(width_mm)
    cur_w, cur_h = fig.get_size_inches()
    if height_mm is None:
        height_in = width_in * (cur_h / cur_w)
    else:
        height_in = mm_to_inch(height_mm)
    fig.set_size_inches(width_in, height_in)
    # fit content into the fixed canvas instead of letting bbox_inches="tight"
    # change the physical size; constrained layout rearranges axes at fixed figsize.
    try:
        fig.set_layout_engine("constrained")
    except Exception:
        try:
            fig.tight_layout()
        except Exception:
            pass
    if formats is None:
        formats = spec["preferred_formats"][:2]
    if dpi is None:
        dpi = spec["min_dpi_line"]
    # key: bbox_inches=None makes the written width exactly equal to the
    # set_size_inches value - the exact-column-width guarantee.
    kwargs.setdefault("bbox_inches", None)
    font_check = _check_font_availability()
    if font_check["warning"]:
        print("[WARNING] " + font_check["warning"], file=sys.stderr)
    written = save_publication_figure(fig, basename, formats=formats, dpi=dpi, **kwargs)
    info = {
        "journal": j,
        "column": column,
        "width_mm": width_mm,
        "height_mm": round(inch_to_mm(height_in), 1),
        "dpi": dpi,
        "formats": list(formats),
        "min_font_pt": spec["min_font_pt"],
        "verified": spec["verified"],
        "note": spec["note"],
        "font": font_check,
    }
    if spec.get("column_caveat"):
        info["column_caveat"] = spec["column_caveat"]
        print("[NOTE] " + spec["column_caveat"], file=sys.stderr)
    return written, info


def check_figure_size(
    fig,
    max_width_mm=None,
    journal=None,
    column="single",
    tol_mm=0.5,
    verbose=True,
    measured=False,
    path=None,
    custom_width_mm=None,
):
    """Verify the figure's physical width (mm) against the column spec.

    With journal, that journal's column width is the limit; else max_width_mm.
    journal="custom" (with custom_width_mm) uses custom_width_mm as the target
      without enforcing a minimum font size.
    Also checks visible text sizes against the journal floor, if defined.

    measured=True: do not trust fig.get_size_inches() (pre-crop canvas size);
      read back the written file's true physical width instead - this is what
      catches bbox_inches="tight" silently changing the column width.
      Needs path (an already written file).
      PNG via PIL pixel width / dpi; PDF/SVG via pypdf when available.
    """
    if journal is not None and journal.lower() == "custom":
        if custom_width_mm is None:
            raise ValueError("journal='custom' requires custom_width_mm (mm)")
        max_width_mm = float(custom_width_mm)
        journal = None  # follow the max_width_mm path; no JOURNAL_SPECS lookup
    if measured:
        if path is None:
            raise ValueError("measured=True requires path (a written file)")
        return _check_measured(
            path,
            max_width_mm=max_width_mm,
            journal=journal,
            column=column,
            tol_mm=tol_mm,
            verbose=verbose,
        )
    w_in, h_in = fig.get_size_inches()
    w_mm, h_mm = inch_to_mm(w_in), inch_to_mm(h_in)
    report = {
        "width_mm": round(w_mm, 2),
        "height_mm": round(h_mm, 2),
        "ok": True,
        "problems": [],
    }
    limit = max_width_mm
    min_font = None
    if journal is not None:
        spec = JOURNAL_SPECS[journal.lower()]
        limit = spec.get(f"{column}_mm", limit)
        min_font = spec["min_font_pt"]
        report["journal"] = journal.lower()
        report["column"] = column
    if limit is not None:
        report["max_width_mm"] = limit
        if w_mm > limit + tol_mm:
            report["ok"] = False
            report["problems"].append(f"width {w_mm:.1f} mm exceeds limit {limit} mm")
    if min_font is not None:
        tiny = _collect_small_fonts(fig, min_font)
        report["min_font_pt"] = min_font
        if tiny:
            report["ok"] = False
            report["problems"].append(
                f"{len(tiny)} text items < {min_font} pt (smallest {min(tiny):.1f} pt)"
            )
    if verbose:
        status = "OK" if report["ok"] else "FAIL"
        print(f"[check_figure_size] {status}  {report}")
    return report


def _collect_small_fonts(fig, min_font_pt):
    """Collect visible text sizes (pt) below the floor."""
    small = []
    for txt in fig.findobj(plt.Text):
        try:
            s = txt.get_text()
        except Exception:
            continue
        if not s or not s.strip():
            continue
        if not txt.get_visible():
            continue
        fs = txt.get_fontsize()
        if fs < min_font_pt - 1e-6:
            small.append(fs)
    return small


def check_scaled_fonts(
    fig,
    journal=None,
    column="single",
    target_width_mm=None,
    min_font_pt=None,
    verbose=True,
):
    """Check effective font sizes after draw-large-then-scale workflows.

    Common workflow: draw on a large canvas (e.g. 180 mm wide), then scale the
    whole figure to the column width (e.g. 89 mm). Every text size shrinks by
    the same factor - check_figure_size only sees the original sizes and would
    miss this. This function covers that blind spot.

    With journal, the column width and min_font_pt are looked up automatically.
    Returns scale, per-item effective sizes, and any violations.

    Only meaningful for the scale-down workflow; with save_for_journal()
    (scale ~ 1) it reports no risk by design - not a bug.
    """
    if journal is not None:
        spec = JOURNAL_SPECS[journal.lower()]
        if target_width_mm is None:
            target_width_mm = spec.get(f"{column}_mm")
        if min_font_pt is None:
            min_font_pt = spec["min_font_pt"]
    if target_width_mm is None or min_font_pt is None:
        raise ValueError("need journal, or both target_width_mm and min_font_pt")

    cur_w_mm = inch_to_mm(fig.get_size_inches()[0])
    scale = target_width_mm / cur_w_mm if cur_w_mm else 1.0
    report = {
        "current_width_mm": round(cur_w_mm, 2),
        "target_width_mm": round(target_width_mm, 2),
        "scale": round(scale, 4),
        "min_font_pt": min_font_pt,
        "ok": True,
        "violations": [],
    }

    if scale >= 0.999:
        # no shrink (or upscale): no risk
        report["note"] = "canvas not shrunk below the column width; no scaling risk"
    for txt in fig.findobj(plt.Text):
        try:
            s = txt.get_text()
        except Exception:
            continue
        if not s or not s.strip() or not txt.get_visible():
            continue
        eff = txt.get_fontsize() * scale
        if eff < min_font_pt - 1e-6:
            report["ok"] = False
            report["violations"].append(
                {
                    "text": s[:30],
                    "raw_pt": round(txt.get_fontsize(), 1),
                    "effective_pt": round(eff, 2),
                }
            )
    if verbose:
        status = "OK" if report["ok"] else "FAIL"
        msg = (
            f"[check_scaled_fonts] {status}  scale={report['scale']} "
            f"({report['current_width_mm']}→{report['target_width_mm']}mm)"
        )
        if report["violations"]:
            worst = min(v["effective_pt"] for v in report["violations"])
            msg += f"  {len(report['violations'])} items < {min_font_pt} pt after scaling (smallest {worst} pt)"
        print(msg)
    return report


def _measure_file_width_mm(path):
    """Read back the true physical width (mm) of a written file.

    Returns (width_mm, backend) or (None, reason). No heavy deps:
      .png  -> PIL (pixel width / dpi) * 25.4
      .pdf  -> pypdf.PdfReader MediaBox (pt = 1/72 in)
      .svg  -> root <svg width="..."> (mm/in/px/pt)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        try:
            from PIL import Image
        except Exception:
            return None, "PIL-missing"
        with Image.open(path) as im:
            px_w = im.size[0]
            dpi = im.info.get("dpi", (None, None))[0]
        if not dpi:
            return None, "png-no-dpi"
        return px_w / float(dpi) * MM_PER_INCH, "PIL"
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception:
            try:
                from PyPDF2 import PdfReader
            except Exception:
                return None, "pypdf-missing"
        box = PdfReader(path).pages[0].mediabox
        return float(box.width) / 72.0 * MM_PER_INCH, "pypdf"
    if ext == ".svg":
        return _measure_svg_width_mm(path)
    return None, f"unsupported-ext-{ext}"


def _measure_svg_width_mm(path):
    """Parse the SVG root width attribute to mm. No deps."""
    import re
    import xml.etree.ElementTree as ET

    try:
        root = ET.parse(path).getroot()
    except Exception as e:
        return None, f"svg-parse-{e.__class__.__name__}"
    raw = root.get("width")
    if not raw:
        return None, "svg-no-width"
    m = re.match(r"\s*([0-9.]+)\s*([a-z%]*)", raw)
    if not m:
        return None, "svg-bad-width"
    val, unit = float(m.group(1)), m.group(2)
    factor = {
        "mm": 1.0,
        "cm": 10.0,
        "in": MM_PER_INCH,
        "pt": MM_PER_INCH / 72.0,
        "px": MM_PER_INCH / 96.0,
        "": MM_PER_INCH / 96.0,
    }.get(unit)
    if factor is None:
        return None, f"svg-unit-{unit}"
    return val * factor, "svg"


def _check_measured(
    path, max_width_mm=None, journal=None, column="single", tol_mm=0.5, verbose=True
):
    """Read the written file's true width and compare with the target column."""
    w_mm, backend = _measure_file_width_mm(path)
    report = {
        "path": path,
        "measured": True,
        "backend": backend,
        "ok": True,
        "problems": [],
    }
    limit = max_width_mm
    if journal is not None:
        spec = JOURNAL_SPECS[journal.lower()]
        limit = spec.get(f"{column}_mm", limit)
        report["journal"] = journal.lower()
        report["column"] = column
    if w_mm is None:
        report["skipped"] = True
        report["reason"] = backend
        if verbose:
            print(f"[check_figure_size:measured] SKIP ({backend})  {report}")
        return report
    report["width_mm"] = round(w_mm, 2)
    if limit is not None:
        report["target_mm"] = limit
        # exact column width: measured must equal the target (two-sided tolerance)
        if abs(w_mm - limit) > tol_mm:
            report["ok"] = False
            report["problems"].append(
                f"measured width {w_mm:.2f} mm deviates from target {limit} mm "
                f"(by {w_mm - limit:+.2f} mm, tolerance {tol_mm} mm)"
            )
    if verbose:
        status = "OK" if report["ok"] else "FAIL"
        print(f"[check_figure_size:measured] {status}  {report}")
    return report


def check_export_compliance(path, journal, column="single", verbose=False):
    """Compliance checks on the written file: measured dpi >= min_dpi, file size
    <= max_file_mb, height <= max_height_px, format in preferred_formats.
    Only judges readable dimensions; unreadable ones are marked skip.
    Returns {ok, problems, checks}."""
    import os as _os

    j = journal.lower()
    if j not in JOURNAL_SPECS:
        return {
            "ok": True,
            "skipped": f"custom/unknown journal {journal}: no compliance spec to check",
            "problems": [],
        }
    spec = JOURNAL_SPECS[j]
    problems, checks = [], {}
    ext = _os.path.splitext(path)[1].lstrip(".").lower()

    # 1) format within preferred_formats
    pref = spec.get("preferred_formats", ())
    checks["format"] = {"ext": ext, "preferred": list(pref)}
    if pref and ext not in pref:
        problems.append(f"format .{ext} not in {j} preferred {pref} (may require conversion)")

    # 2) file size
    mb = _os.path.getsize(path) / 1e6 if _os.path.exists(path) else None
    if mb is not None:
        checks["file_mb"] = round(mb, 2)
        cap = spec.get("max_file_mb")
        if cap and mb > cap:
            problems.append(f"file {mb:.1f} MB exceeds {j} cap {cap} MB")

    # 3) bitmap dpi and height (readable for PNG/TIFF; vector skipped)
    if ext in ("png", "tiff", "tif"):
        try:
            from PIL import Image

            with Image.open(path) as im:
                w_px, h_px = im.size
                dpi = im.info.get("dpi", (None, None))[0]
            checks["pixels"] = {"w": w_px, "h": h_px, "dpi": dpi}
            min_dpi = spec.get("min_dpi_halftone") or spec.get("min_dpi_line")
            if dpi and min_dpi and dpi < min_dpi - 1:
                problems.append(f"measured dpi={dpi} < {j} floor {min_dpi}")
            max_h = spec.get("max_height_px")
            if max_h and h_px > max_h:
                problems.append(f"height {h_px} px exceeds {j} cap {max_h} px")
        except ImportError:
            checks["pixels"] = "skip (no Pillow)"
        except Exception as e:
            checks["pixels"] = f"skip({e.__class__.__name__})"
    else:
        checks["pixels"] = f"skip (vector / unreadable dpi: .{ext})"

    report = {
        "path": path,
        "journal": j,
        "ok": len(problems) == 0,
        "problems": problems,
        "checks": checks,
    }
    if verbose:
        print(f"[check_export_compliance] {'OK' if report['ok'] else 'FAIL'}  {report}")
    return report


def _demo_and_selfcheck():
    """Produce a demo figure, run the functions, assert key invariants."""
    import numpy as np

    here = os.path.dirname(os.path.abspath(__file__))
    style = os.path.join(here, "..", "assets", "publication.mplstyle")
    if os.path.exists(style):
        plt.style.use(style)
    x = np.linspace(0, 2 * np.pi, 100)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(x, np.sin(x), label="sin")
    ax.plot(x, np.cos(x), label="cos", linestyle="--")
    ax.fill_between(x, np.sin(x) - 0.1, np.sin(x) + 0.1, alpha=0.2)
    ax.set_xlabel("phase (rad)")
    ax.set_ylabel("amplitude (a.u.)")
    ax.set_title("a", loc="left")
    ax.legend()

    outdir = os.path.join(here, "..", "examples", "_export_demo")
    base = os.path.join(outdir, "demo_export")
    written = save_publication_figure(fig, base, formats=("pdf", "png", "svg"))
    assert all(os.path.exists(p) and os.path.getsize(p) > 0 for p in written), written
    print("[demo] save_publication_figure ->", [os.path.basename(p) for p in written])

    # export at Nature single column and verify
    fig2, ax2 = plt.subplots(figsize=(5.0, 4.0))
    ax2.bar(["A", "B", "C"], [3, 5, 2])
    ax2.set_ylabel("count")
    nat_base = os.path.join(outdir, "demo_nature")
    w2, info = save_for_journal(
        fig2, nat_base, journal="nature", column="single", formats=("pdf", "png")
    )
    assert abs(info["width_mm"] - 89.0) < 0.6, info
    print(
        "[demo] save_for_journal nature/single ->",
        info["width_mm"],
        "mm",
        [os.path.basename(p) for p in w2],
    )

    # key regression: read back the written file's true width rather than the
    # set value; bbox_inches="tight" used to silently change the width here.
    for fmt in ("png", "pdf"):
        fpath = f"{nat_base}.{fmt}"
        rep_m = check_figure_size(
            fig2,
            journal="nature",
            column="single",
            measured=True,
            path=fpath,
            tol_mm=0.5,
        )
        if rep_m.get("skipped"):
            print(f"[demo] measured {fmt}: SKIP ({rep_m['reason']}) - no read backend")
            continue
        assert rep_m["ok"], rep_m
        assert abs(rep_m["width_mm"] - 89.0) < 0.5, rep_m
        print(
            f"[demo] read-back width ({fmt})={rep_m['width_mm']} mm "
            f"== target 89.0 mm  [backend={rep_m['backend']}] PASS"
        )

    rep_ok = check_figure_size(fig2, journal="nature", column="single", verbose=False)
    assert rep_ok["ok"], rep_ok

    # font fallback check: info carries a font dict (fell_back=True + warning when no Arial)
    assert "font" in info and "fell_back" in info["font"], info.get("font")
    if info["font"]["fell_back"]:
        assert info["font"]["warning"], "expected warning on fallback"
        print(f"[demo] font check: fell back to '{info['font']['resolved']}' with warning PASS")
    else:
        print(f"[demo] font check: preferred '{info['font']['resolved']}' available PASS")

    # export compliance: dpi/size/format/height fields
    png_path = f"{nat_base}.png"
    if os.path.exists(png_path):
        comp = check_export_compliance(png_path, "nature", "single")
        assert "checks" in comp and "format" in comp["checks"], comp
        print(
            f"[demo] check_export_compliance(nature png): ok={comp['ok']} "
            f"problems={comp['problems'][:1]}"
        )
    # plos only accepts tiff/eps; png should fail the format check
    if os.path.exists(png_path):
        comp_plos = check_export_compliance(png_path, "plos", "single")
        assert any("format" in p for p in comp_plos["problems"]), comp_plos
        print("[demo] check_export_compliance(plos, png): format violation detected PASS")
    # custom/unknown journal skips compliance without crashing
    assert (
        check_export_compliance(png_path, "custom").get("skipped")
        if os.path.exists(png_path)
        else True
    )

    # deliberately oversized figure -> should FAIL
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot([0, 1], [0, 1])
    rep_bad = check_figure_size(fig3, journal="nature", column="single", verbose=False)
    assert not rep_bad["ok"], rep_bad
    print("[demo] check_figure_size: compliant OK, oversized correctly FAIL")

    # MDPI single-column (170 mm) export with read-back verification
    fig4, ax4 = plt.subplots(figsize=(6.0, 4.0))
    ax4.plot([0, 1, 2], [1, 3, 2])
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    mdpi_base = os.path.join(outdir, "demo_mdpi")
    w4, info4 = save_for_journal(
        fig4, mdpi_base, journal="mdpi", column="single", formats=("pdf", "png")
    )
    assert abs(info4["width_mm"] - 170.0) < 0.6, info4
    print("[demo] save_for_journal mdpi/single ->", info4["width_mm"], "mm")

    # custom_width_mm escape hatch (e.g. an 84 mm text-column journal)
    fig5, ax5 = plt.subplots(figsize=(5.0, 3.5))
    ax5.bar(["A", "B"], [2, 4])
    cust_base = os.path.join(outdir, "demo_custom")
    w5, info5 = save_for_journal(
        fig5, cust_base, journal="custom", custom_width_mm=84.0, formats=("pdf", "png")
    )
    assert abs(info5["width_mm"] - 84.0) < 0.6, info5
    rep5 = check_figure_size(
        fig5, journal="custom", custom_width_mm=84.0, verbose=False
    )
    assert rep5["ok"] and rep5["max_width_mm"] == 84.0, rep5
    print(
        "[demo] custom_width_mm=84 ->",
        info5["width_mm"],
        "mm, check OK",
    )

    plt.close("all")
    print("[selfcheck] ALL PASS, output dir:", os.path.abspath(outdir))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--selftest":
        raise SystemExit("usage: python figure_export.py [--selftest]")
    _demo_and_selfcheck()
