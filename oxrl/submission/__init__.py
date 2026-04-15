"""
oxrl.submission -- NeurIPS paper submission compliance toolkit.

Automates the mechanical parts of preparing a NeurIPS submission:
  - Croissant metadata generation with RAI fields + validation
  - Paper PDF compliance checks (checklist, anonymization, page limits)
  - README reproduction section generation
  - HuggingFace asset inventory

Modules:
    croissant:   Generate and validate Croissant metadata for HF datasets.
    paper_check: Audit a LaTeX source file for NeurIPS compliance.
    readme_gen:  Generate a reproduction section for README.md.
    run_submit:  CLI entry point tying everything together.
"""
from oxrl.submission.croissant import generate_croissant, validate_croissant
from oxrl.submission.paper_check import check_paper
from oxrl.submission.readme_gen import generate_reproduction_section
