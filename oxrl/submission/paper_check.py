"""
NeurIPS paper compliance checker.

Audits a LaTeX source file for the mechanical requirements that cause
desk rejections: missing checklist, broken anonymization, page limit
violations, and wrong style file.
"""
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────
# Individual checks
# ──────────────────────────────────────────────────────────────────────

def check_anonymization(tex: str) -> List[Dict]:
    """Check double-blind anonymization requirements."""
    issues = []

    # [final] option reveals author names.
    if re.search(r"\\usepackage\[.*final.*\]\{neurips_20\d\d\}", tex):
        issues.append({
            "severity": "CRITICAL",
            "check": "anonymization",
            "message": (
                "\\usepackage[final]{neurips_20xx} removes anonymization. "
                "Remove the [final] option for review submission."
            ),
        })

    # Author field should say Anonymous or be blank.
    author_match = re.search(
        r"\\author\{(.*?)\}", tex, re.DOTALL
    )
    if author_match:
        author_text = author_match.group(1).strip()
        if author_text and "anonymous" not in author_text.lower():
            issues.append({
                "severity": "CRITICAL",
                "check": "anonymization",
                "message": (
                    f"Author field contains: '{author_text[:80]}'. "
                    "Must be blank or 'Anonymous Authors' for double-blind."
                ),
            })

    # Self-revealing citations.
    revealing_patterns = [
        (r"\b[Oo]ur\s+(previous|prior|earlier)\s+work\b", "self-citation: 'our previous work'"),
        (r"\b[Ww]e\s+(previously|earlier)\s+", "self-citation: 'we previously'"),
        (r"\b[Ii]n\s+our\s+work\b", "self-citation: 'in our work'"),
    ]
    for pattern, desc in revealing_patterns:
        if re.search(pattern, tex):
            issues.append({
                "severity": "WARNING",
                "check": "anonymization",
                "message": f"Potential de-anonymization via {desc}",
            })

    return issues


def check_checklist(tex: str) -> List[Dict]:
    """Check that the NeurIPS paper checklist is present."""
    issues = []

    has_checklist = bool(
        re.search(r"\\section\*?\{.*[Cc]hecklist.*\}", tex)
        or re.search(r"\\neuripscheck", tex)
    )
    if not has_checklist:
        issues.append({
            "severity": "CRITICAL",
            "check": "checklist",
            "message": (
                "Paper checklist section not found. NeurIPS requires a "
                "'Paper Checklist' section after references and before "
                "appendices. This will cause a desk rejection."
            ),
        })
        return issues

    # Check checklist placement: should be after references, before appendix.
    bib_pos = _find_last(tex, r"\\end\{thebibliography\}")
    if bib_pos is None:
        bib_pos = _find_last(tex, r"\\bibliography\{")
    checklist_pos = _find_first(tex, r"\\section\*?\{.*[Cc]hecklist.*\}")
    appendix_pos = _find_first(tex, r"\\appendix")

    if checklist_pos is not None and bib_pos is not None:
        if checklist_pos < bib_pos:
            issues.append({
                "severity": "WARNING",
                "check": "checklist",
                "message": "Checklist appears before references. It should come after.",
            })

    if checklist_pos is not None and appendix_pos is not None:
        if checklist_pos > appendix_pos:
            issues.append({
                "severity": "WARNING",
                "check": "checklist",
                "message": "Checklist appears after \\appendix. It should come before.",
            })

    # Check that answer commands exist.
    if "\\answerYes" not in tex and "\\answerNo" not in tex:
        issues.append({
            "severity": "WARNING",
            "check": "checklist",
            "message": (
                "No \\answerYes/\\answerNo/\\answerNA commands found. "
                "Checklist items should use these answer macros."
            ),
        })

    return issues


def check_style(tex: str) -> List[Dict]:
    """Check NeurIPS style file usage."""
    issues = []

    style_match = re.search(r"\\usepackage(?:\[.*?\])?\{neurips_(\d{4})\}", tex)
    if not style_match:
        issues.append({
            "severity": "CRITICAL",
            "check": "style",
            "message": "No neurips_YYYY style file detected. Use \\usepackage{neurips_2026}.",
        })
    else:
        year = int(style_match.group(1))
        if year < 2026:
            issues.append({
                "severity": "WARNING",
                "check": "style",
                "message": (
                    f"Using neurips_{year}.sty — expected neurips_2026.sty. "
                    "Margin or font requirements may differ between years."
                ),
            })

    return issues


def check_page_limit(tex_path: str) -> List[Dict]:
    """Compile and check main-text page count (≤9 pages)."""
    issues = []
    tex_path = Path(tex_path)

    if not tex_path.exists():
        issues.append({
            "severity": "ERROR",
            "check": "pages",
            "message": f"File not found: {tex_path}",
        })
        return issues

    pdf_path = tex_path.with_suffix(".pdf")
    pdftotext = _which("pdftotext")

    # Try to compile if no PDF or PDF is older than tex.
    if not pdf_path.exists() or pdf_path.stat().st_mtime < tex_path.stat().st_mtime:
        pdflatex = _which("pdflatex")
        if pdflatex:
            subprocess.run(
                [pdflatex, "-interaction=nonstopmode", tex_path.name],
                cwd=str(tex_path.parent),
                capture_output=True,
                timeout=120,
            )

    if not pdf_path.exists():
        issues.append({
            "severity": "WARNING",
            "check": "pages",
            "message": "Could not compile PDF — install pdflatex to check page limits.",
        })
        return issues

    # Count total pages.
    total_pages = _count_pdf_pages(pdf_path)

    # Find where references start to determine main-text length.
    if pdftotext:
        main_pages = _find_references_page(pdf_path, pdftotext)
        if main_pages and main_pages > 9:
            issues.append({
                "severity": "CRITICAL",
                "check": "pages",
                "message": (
                    f"Main text appears to be {main_pages} pages (limit: 9). "
                    "References start on page {main_pages + 1}."
                ),
            })
        elif main_pages:
            issues.append({
                "severity": "INFO",
                "check": "pages",
                "message": f"Main text: {main_pages} pages, total: {total_pages} pages.",
            })
    else:
        issues.append({
            "severity": "INFO",
            "check": "pages",
            "message": (
                f"Total PDF: {total_pages} pages. "
                "Install pdftotext to verify main-text page count."
            ),
        })

    return issues


def check_pdf_metadata(tex_path: str) -> List[Dict]:
    """Check compiled PDF for author metadata leaks."""
    issues = []
    pdf_path = Path(tex_path).with_suffix(".pdf")

    if not pdf_path.exists():
        return issues

    pdfinfo = _which("pdfinfo")
    if not pdfinfo:
        issues.append({
            "severity": "INFO",
            "check": "pdf_metadata",
            "message": "pdfinfo not available — cannot check PDF metadata.",
        })
        return issues

    try:
        result = subprocess.run(
            [pdfinfo, str(pdf_path)],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if line.lower().startswith("author:"):
                author = line.split(":", 1)[1].strip()
                if author and author.lower() not in ("", "anonymous"):
                    issues.append({
                        "severity": "WARNING",
                        "check": "pdf_metadata",
                        "message": (
                            f"PDF Author metadata contains: '{author}'. "
                            "Right-click PDF → Properties to verify."
                        ),
                    })
    except Exception:
        pass

    return issues


def check_content_accuracy(tex: str) -> List[Dict]:
    """Check that key checklist-relevant content is present."""
    issues = []

    if not re.search(r"[Ll]imitation", tex):
        issues.append({
            "severity": "WARNING",
            "check": "content",
            "message": "No limitations discussion found. Reviewers expect this.",
        })

    if not re.search(r"GPU.?hour|compute.*budget|compute.*cost", tex, re.IGNORECASE):
        issues.append({
            "severity": "WARNING",
            "check": "content",
            "message": (
                "No compute budget information found. "
                "Include total GPU-hours for transparency."
            ),
        })

    if not re.search(r"seed|\\pm|\\sigma|standard deviation|error bar", tex, re.IGNORECASE):
        issues.append({
            "severity": "WARNING",
            "check": "content",
            "message": "No mention of seeds/error bars/std. Report statistical significance.",
        })

    return issues


# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────

def check_paper(tex_path: str) -> Dict:
    """Run all compliance checks on a LaTeX source file.

    Returns a dict with:
        passed: bool — True if no CRITICAL issues found
        issues: list of dicts with severity/check/message
        summary: dict mapping check name to pass/fail
    """
    tex_path = Path(tex_path)
    tex = tex_path.read_text()

    all_issues = []
    all_issues.extend(check_anonymization(tex))
    all_issues.extend(check_checklist(tex))
    all_issues.extend(check_style(tex))
    all_issues.extend(check_page_limit(str(tex_path)))
    all_issues.extend(check_pdf_metadata(str(tex_path)))
    all_issues.extend(check_content_accuracy(tex))

    checks = {}
    for issue in all_issues:
        name = issue["check"]
        if name not in checks:
            checks[name] = "PASS"
        if issue["severity"] == "CRITICAL":
            checks[name] = "FAIL"
        elif issue["severity"] in ("WARNING", "ERROR") and checks[name] != "FAIL":
            checks[name] = "WARN"

    has_critical = any(i["severity"] == "CRITICAL" for i in all_issues)

    return {
        "passed": not has_critical,
        "issues": all_issues,
        "summary": checks,
    }


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _find_first(text: str, pattern: str) -> Optional[int]:
    m = re.search(pattern, text)
    return m.start() if m else None


def _find_last(text: str, pattern: str) -> Optional[int]:
    matches = list(re.finditer(pattern, text))
    return matches[-1].start() if matches else None


def _which(cmd: str) -> Optional[str]:
    import shutil
    return shutil.which(cmd)


def _count_pdf_pages(pdf_path: Path) -> Optional[int]:
    pdfinfo = _which("pdfinfo")
    if not pdfinfo:
        return None
    try:
        result = subprocess.run(
            [pdfinfo, str(pdf_path)], capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    return None


def _find_references_page(pdf_path: Path, pdftotext: str) -> Optional[int]:
    """Find the last page before References/Bibliography starts."""
    try:
        total = _count_pdf_pages(pdf_path)
        if not total:
            return None
        for page in range(1, min(total + 1, 20)):
            result = subprocess.run(
                [pdftotext, "-f", str(page), "-l", str(page), str(pdf_path), "-"],
                capture_output=True, text=True, timeout=10,
            )
            text = result.stdout
            if re.search(r"^\s*References\s*$", text, re.MULTILINE):
                return page - 1
    except Exception:
        pass
    return None
