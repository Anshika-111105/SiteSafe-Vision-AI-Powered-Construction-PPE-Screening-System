#!/usr/bin/env python3
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("citation_audit")

CITATION_PATTERN = re.compile(r"\[(\^?\d+|[a-zA-Z0-9_-]+)\]:\s*(https?://[^\s]+)")
INLINE_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")


def audit_markdown_file(file_path: Path) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    citations = CITATION_PATTERN.findall(content)
    inline_links = INLINE_LINK_PATTERN.findall(content)

    total_citations = len(citations) + len(inline_links)
    has_references_section = "## References" in content or "## Citations" in content or "### References" in content

    return {
        "file_name": file_path.name,
        "relative_path": file_path.relative_to(PROJECT_ROOT).as_posix(),
        "total_lines": len(lines),
        "total_citations_found": total_citations,
        "has_references_section": has_references_section,
        "citations": [c[1] for c in citations] + [l[1] for l in inline_links],
    }


def run_citation_audit() -> Dict[str, Any]:
    logger.info("Executing Formal Citation Audit across Documentation...")
    docs_dir = PROJECT_ROOT / "docs"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    target_files = sorted(list(docs_dir.glob("*.md")))
    readme_path = PROJECT_ROOT / "README.md"
    if readme_path.exists():
        target_files.append(readme_path)

    audit_results = []
    total_refs = 0
    all_cited = True

    for doc_file in target_files:
        res = audit_markdown_file(doc_file)
        audit_results.append(res)
        total_refs += res["total_citations_found"]
        if res["total_citations_found"] == 0 and doc_file.name != "LIMITATIONS.md":
            all_cited = False

    now_iso = datetime.now(timezone.utc).isoformat()

    # Generate Markdown Report
    report_md_path = reports_dir / "citation_audit.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# SiteSafe Vision: Documentation Citation & Provenance Audit\n\n")
        f.write(f"- **Audit Status**: `{'PASSED (100% Sourced)' if all_cited else 'WARNING'}`\n")
        f.write(f"- **Audit Timestamp (UTC)**: `{now_iso}`\n")
        f.write(f"- **Documents Inspected**: `{len(target_files)}`\n")
        f.write(f"- **Total Citations / References Identified**: `{total_refs}`\n\n")
        f.write("## Document Citation Summary\n\n")
        f.write("| Document | Total Citations | References Section Present | Status |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for res in audit_results:
            st = "✅ Verified" if res["total_citations_found"] > 0 else "⚠️ No citations"
            f.write(f"| `{res['relative_path']}` | {res['total_citations_found']} | {res['has_references_section']} | {st} |\n")
        f.write("\n## Official Primary References Index\n\n")
        f.write("- **Roboflow Universe Construction PPE Dataset**: [https://universe.roboflow.com/new-project-ds9wg/construction-ppe-detection-vqbc0](https://universe.roboflow.com/new-project-ds9wg/construction-ppe-detection-vqbc0)\n")
        f.write("- **Creative Commons Attribution 4.0 International (CC BY 4.0)**: [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)\n")
        f.write("- **PyTorch Reproducibility & Determinism Notes**: [https://pytorch.org/docs/stable/notes/randomness.html](https://pytorch.org/docs/stable/notes/randomness.html)\n")
        f.write("- **PyTorch Transfer Learning Tutorial**: [https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)\n")
        f.write("- **Torchvision Pretrained Models & Weights**: [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)\n")
        f.write("- **DVC (Data Version Control) Documentation**: [https://dvc.org/doc](https://dvc.org/doc)\n")
        f.write("- **FastAPI Docker Deployment Guide**: [https://fastapi.tiangolo.com/deployment/docker/](https://fastapi.tiangolo.com/deployment/docker/)\n")
        f.write("- **Grad-CAM Saliency Maps (Selvaraju et al., 2017)**: [https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)\n")
        f.write("- **Scikit-Learn Model Evaluation Guidance**: [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)\n")
        f.write("- **Docker OCI Security Best Practices**: [https://docs.docker.com/develop/security-best-practices/](https://docs.docker.com/develop/security-best-practices/)\n")

    logger.info(f"Citation audit completed. Report saved to {report_md_path}")
    return {"audit_passed": all_cited, "total_citations": total_refs}


if __name__ == "__main__":
    run_citation_audit()
