"""
NeurIPS paper checklist generator.

Generates the LaTeX for the 16-item NeurIPS paper checklist with
pre-filled answers based on project metadata.
"""
from typing import Dict, List, Optional


# The 16 NeurIPS checklist questions.
CHECKLIST_QUESTIONS = [
    {
        "id": 1,
        "title": "Claims",
        "question": (
            "Do the main claims made in the abstract and introduction "
            "accurately reflect the paper's contributions and scope?"
        ),
    },
    {
        "id": 2,
        "title": "Limitations",
        "question": "Does the paper discuss the limitations of the work performed by the authors?",
    },
    {
        "id": 3,
        "title": "Theory, Assumptions and Proofs",
        "question": (
            "For each theoretical result, does the paper provide the full "
            "set of assumptions and a complete proof?"
        ),
    },
    {
        "id": 4,
        "title": "Experimental Result Reproducibility",
        "question": (
            "Does the paper fully disclose all the information needed to "
            "reproduce the main experimental results of the paper to the "
            "extent that it affects the main claims and/or conclusions of the paper?"
        ),
    },
    {
        "id": 5,
        "title": "Open Access to Data and Code",
        "question": (
            "Does the paper provide open access to the data and code, with "
            "sufficient instructions to faithfully reproduce the main experimental results?"
        ),
    },
    {
        "id": 6,
        "title": "Experimental Setting/Details",
        "question": (
            "Does the paper specify all the training and test details "
            "necessary to understand the results?"
        ),
    },
    {
        "id": 7,
        "title": "Experiment Statistical Significance",
        "question": (
            "Does the paper report error bars suitably and provide measures "
            "of statistical significance?"
        ),
    },
    {
        "id": 8,
        "title": "Experiments Compute Resource",
        "question": (
            "For each experiment, does the paper provide sufficient "
            "information about compute resources?"
        ),
    },
    {
        "id": 9,
        "title": "Code Of Ethics",
        "question": (
            "Have the authors read the NeurIPS Code of Ethics and ensured "
            "that the paper conforms to it?"
        ),
    },
    {
        "id": 10,
        "title": "Broader Impacts",
        "question": (
            "Does the paper discuss both potential positive societal impacts "
            "and negative societal impacts of the work?"
        ),
    },
    {
        "id": 11,
        "title": "Safeguards",
        "question": (
            "Does the paper describe safeguards that have been put in place "
            "for responsible disclosure of data or models with high potential for misuse?"
        ),
    },
    {
        "id": 12,
        "title": "Licenses for Existing Assets",
        "question": (
            "Are the creators of assets used in the paper properly credited "
            "and are the terms of use respected?"
        ),
    },
    {
        "id": 13,
        "title": "New Assets",
        "question": (
            "Are new assets introduced in the paper well documented and is "
            "the documentation provided alongside the assets?"
        ),
    },
    {
        "id": 14,
        "title": "Crowdsourcing and Research with Human Subjects",
        "question": (
            "For crowdsourcing experiments and research with human subjects, "
            "does the paper include the full text of instructions given to participants?"
        ),
    },
    {
        "id": 15,
        "title": "Institutional Review Board (IRB) Approvals",
        "question": "Does the paper describe potential risks incurred by study participants?",
    },
    {
        "id": 16,
        "title": "Declaration of LLM Usage",
        "question": (
            "Does the paper describe the usage of LLMs if they are used as "
            "an important component of the core methodology?"
        ),
    },
]


def generate_checklist_latex(answers: List[Dict]) -> str:
    """Generate LaTeX for the NeurIPS paper checklist.

    Args:
        answers: List of dicts with keys:
            id (int): question number 1-16
            answer (str): "Yes", "No", or "NA"
            justification (str): free text

    Returns:
        LaTeX string for the checklist section.
    """
    answer_map = {a["id"]: a for a in answers}

    lines = [
        "% ================================================================",
        "% PAPER CHECKLIST",
        "% ================================================================",
        "\\section*{Paper Checklist}",
        "",
        "\\begin{enumerate}",
        "",
    ]

    for q in CHECKLIST_QUESTIONS:
        qid = q["id"]
        a = answer_map.get(qid, {})
        answer = a.get("answer", "TODO")
        justification = a.get("justification", "TODO: fill in justification")

        if answer == "Yes":
            answer_cmd = "\\answerYes{}"
        elif answer == "No":
            answer_cmd = "\\answerNo{}"
        elif answer == "NA":
            answer_cmd = "\\answerNA{}"
        else:
            answer_cmd = "\\textcolor{red}{[TODO]}"

        lines.extend([
            f"\\item {{\\bf {q['title']}}}",
            f"    \\item[] Question: {q['question']}",
            f"    \\item[] Answer: {answer_cmd}",
            f"    \\item[] Justification: {justification}",
            "",
        ])

    lines.extend([
        "\\end{enumerate}",
    ])

    return "\n".join(lines)


def generate_answer_macros() -> str:
    """Return LaTeX command definitions for checklist answers."""
    return (
        "% Checklist answer commands\n"
        "\\newcommand{\\answerYes}[1][]{\\textcolor{blue}{[Yes]}#1}\n"
        "\\newcommand{\\answerNo}[1][]{\\textcolor{red}{[No]}#1}\n"
        "\\newcommand{\\answerNA}[1][]{\\textcolor{gray}{[N/A]}#1}\n"
    )
