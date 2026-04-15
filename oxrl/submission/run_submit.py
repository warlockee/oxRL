"""
CLI entry point for NeurIPS submission compliance.

Usage:
    # Run all checks on a paper
    python -m oxrl.submission.run_submit check \
        --paper docs/oxrl_formal.tex \
        --readme README.md

    # Generate Croissant metadata with RAI fields
    python -m oxrl.submission.run_submit croissant \
        --dataset warlockee/oxrl-nips-2026 \
        --output neurips_croissant/croissant.json \
        --hf-token $HF_TOKEN \
        --biases "Describe biases..." \
        --collection "Describe collection..." \
        --pii "No PII." \
        --date 2026-04-13 \
        --version 1.0.0

    # Check README completeness against Papers With Code checklist
    python -m oxrl.submission.run_submit readme \
        --readme README.md

    # Full pre-submission audit (paper + readme + croissant)
    python -m oxrl.submission.run_submit audit \
        --paper docs/oxrl_formal.tex \
        --readme README.md \
        --croissant neurips_croissant/croissant.json
"""
import argparse
import json
import os
import sys
from pathlib import Path


def cmd_check(args):
    """Run paper compliance checks."""
    from oxrl.submission.paper_check import check_paper

    result = check_paper(args.paper)

    print("=" * 60)
    print("NeurIPS Paper Compliance Check")
    print("=" * 60)
    print()

    # Summary.
    for check, status in result["summary"].items():
        icon = {"PASS": "OK", "WARN": "!!", "FAIL": "XX"}[status]
        print(f"  [{icon}] {check}: {status}")
    print()

    # Details.
    for issue in result["issues"]:
        sev = issue["severity"]
        prefix = {"CRITICAL": "XX", "WARNING": "!!", "ERROR": "!!", "INFO": "--"}[sev]
        print(f"  [{prefix}] [{issue['check']}] {issue['message']}")

    print()
    if result["passed"]:
        print("RESULT: PASSED — no critical issues found.")
    else:
        print("RESULT: FAILED — critical issues must be fixed before submission.")

    return 0 if result["passed"] else 1


def cmd_croissant(args):
    """Generate and validate Croissant metadata."""
    from oxrl.submission.croissant import generate_croissant, validate_croissant

    print(f"Fetching Croissant metadata for {args.dataset}...")
    path = generate_croissant(
        dataset_id=args.dataset,
        output_path=args.output,
        hf_token=args.hf_token,
        data_biases=args.biases or "",
        data_collection=args.collection or "",
        personal_sensitive_info=args.pii or "No personal or sensitive information.",
        is_live_dataset=False,
        date_published=args.date,
        version=args.version,
        license_url=args.license_url,
        cite_as=args.cite_as,
    )
    print(f"Wrote: {path}")

    if args.validate:
        print("Validating...")
        result = validate_croissant(str(path))
        if result["errors"]:
            for e in result["errors"]:
                print(f"  [XX] {e}")
        if result["warnings"]:
            for w in result["warnings"]:
                print(f"  [!!] {w}")
        if result["valid"]:
            print("  [OK] Validation passed.")
        else:
            print("  [XX] Validation failed.")
            return 1

    return 0


def cmd_readme(args):
    """Check README against Papers With Code checklist."""
    from oxrl.submission.readme_gen import check_readme_completeness

    result = check_readme_completeness(args.readme)

    print("=" * 60)
    print("Papers With Code — ML Code Completeness Checklist")
    print("=" * 60)
    print()

    all_pass = True
    for item, passed in result.items():
        icon = "OK" if passed else "XX"
        if not passed:
            all_pass = False
        print(f"  [{icon}] {item}")

    print()
    if all_pass:
        print("RESULT: All 5 checklist items satisfied.")
    else:
        print("RESULT: Some items missing — see above.")

    return 0 if all_pass else 1


def cmd_audit(args):
    """Run full pre-submission audit."""
    exit_code = 0

    # Paper check.
    if args.paper:
        print()
        print("=" * 60)
        print(" 1/3  PAPER COMPLIANCE")
        print("=" * 60)
        from oxrl.submission.paper_check import check_paper

        result = check_paper(args.paper)
        for issue in result["issues"]:
            sev = issue["severity"]
            prefix = {"CRITICAL": "XX", "WARNING": "!!", "ERROR": "!!", "INFO": "--"}[sev]
            print(f"  [{prefix}] [{issue['check']}] {issue['message']}")
        if not result["passed"]:
            exit_code = 1
        print()

    # README check.
    if args.readme:
        print("=" * 60)
        print(" 2/3  README COMPLETENESS")
        print("=" * 60)
        from oxrl.submission.readme_gen import check_readme_completeness

        result = check_readme_completeness(args.readme)
        for item, passed in result.items():
            icon = "OK" if passed else "XX"
            if not passed:
                exit_code = 1
            print(f"  [{icon}] {item}")
        print()

    # Croissant check.
    if args.croissant:
        print("=" * 60)
        print(" 3/3  CROISSANT METADATA")
        print("=" * 60)
        from oxrl.submission.croissant import validate_croissant

        result = validate_croissant(args.croissant)
        if result["errors"]:
            for e in result["errors"]:
                print(f"  [XX] {e}")
                exit_code = 1
        if result["warnings"]:
            for w in result["warnings"]:
                print(f"  [!!] {w}")
        if result["valid"]:
            print("  [OK] Croissant valid.")
        print()

    # Final verdict.
    print("=" * 60)
    if exit_code == 0:
        print("AUDIT PASSED — ready to submit.")
    else:
        print("AUDIT FAILED — fix issues above before submission.")
    print("=" * 60)

    return exit_code


def main():
    parser = argparse.ArgumentParser(
        description="oxRL NeurIPS submission compliance toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # check
    p_check = subparsers.add_parser("check", help="Check paper LaTeX for NeurIPS compliance")
    p_check.add_argument("--paper", required=True, help="Path to main .tex file")

    # croissant
    p_crois = subparsers.add_parser("croissant", help="Generate Croissant metadata with RAI fields")
    p_crois.add_argument("--dataset", required=True, help="HuggingFace dataset ID (e.g. user/dataset)")
    p_crois.add_argument("--output", required=True, help="Output JSON path")
    p_crois.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="HuggingFace token")
    p_crois.add_argument("--biases", help="Data biases description")
    p_crois.add_argument("--collection", help="Data collection description")
    p_crois.add_argument("--pii", help="Personal/sensitive info description")
    p_crois.add_argument("--date", help="Date published (YYYY-MM-DD)")
    p_crois.add_argument("--version", help="Dataset version")
    p_crois.add_argument("--license-url", help="License URL")
    p_crois.add_argument("--cite-as", help="BibTeX citation")
    p_crois.add_argument("--validate", action="store_true", default=True, help="Validate after generation")
    p_crois.add_argument("--no-validate", dest="validate", action="store_false")

    # readme
    p_readme = subparsers.add_parser("readme", help="Check README against Papers With Code checklist")
    p_readme.add_argument("--readme", default="README.md", help="Path to README.md")

    # audit
    p_audit = subparsers.add_parser("audit", help="Full pre-submission audit")
    p_audit.add_argument("--paper", help="Path to main .tex file")
    p_audit.add_argument("--readme", help="Path to README.md")
    p_audit.add_argument("--croissant", help="Path to Croissant JSON file")

    args = parser.parse_args()

    if args.command == "check":
        sys.exit(cmd_check(args))
    elif args.command == "croissant":
        sys.exit(cmd_croissant(args))
    elif args.command == "readme":
        sys.exit(cmd_readme(args))
    elif args.command == "audit":
        sys.exit(cmd_audit(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
