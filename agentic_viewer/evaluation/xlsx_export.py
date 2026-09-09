"""Export evaluation and inference results to Excel (.xlsx) format."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from agentic_viewer.evaluation.baseline import load_or_compute_run_eval
from agentic_viewer.evaluation.summary import build_evaluation_summary, read_agentic_evals
from agentic_viewer.pdf_source import infer_run_document

EXCEL_COLUMNS = [
    "파일명",
    "Key",
    "Prediction",
    "Ground Truth",
    "채점결과",
    "검색페이지",
    "추출근거",
    "검색근거",
    "검토의견-요약",
    "검토의견-상세",
]

PER_RUN_COLUMNS = [
    "Run ID",
    "파일명",
    "Value EM",
    "Page F1",
    "Evid F1",
    "Agentic Done",
    "Pred Acc",
    "GT Valid",
]


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _format_search_pages(pages_val: Any) -> str:
    if not pages_val:
        return ""
    if isinstance(pages_val, (list, tuple, set)):
        sorted_pages = []
        for p in pages_val:
            try:
                sorted_pages.append((int(p), str(p)))
            except (ValueError, TypeError):
                sorted_pages.append((999999, str(p)))
        sorted_pages.sort(key=lambda x: x[0])
        return ", ".join(x[1] for x in sorted_pages)
    return str(pages_val)


def _format_search_reasons(reasons_val: Any) -> str:
    if not reasons_val:
        return ""
    if isinstance(reasons_val, dict):
        lines = []
        items = []
        for k, v in reasons_val.items():
            try:
                page_num = int(k)
            except (ValueError, TypeError):
                page_num = 999999
            items.append((page_num, str(k), str(v or "").strip()))
        items.sort(key=lambda x: x[0])
        for _, k_str, v_str in items:
            if v_str:
                lines.append(f"p.{k_str}: {v_str}")
        return "\n".join(lines)
    return str(reasons_val).strip()


def _format_verdict(is_correct: Optional[str], exact_match: Optional[bool], status: Optional[str]) -> str:
    if is_correct:
        norm = str(is_correct).strip().lower()
        if norm == "correct":
            return "정답"
        if norm == "incorrect":
            return "오답"
        return str(is_correct).strip()

    if status == "running":
        return "평가중"

    if exact_match is True:
        return "정답 (EM)"
    if exact_match is False:
        return "오답 (EM)"

    return "미채점"


def extract_run_rows(run_dir: Path) -> List[Dict[str, str]]:
    """Extract evaluation and extraction rows for a single run directory."""
    if not run_dir.is_dir():
        return []

    doc_name = infer_run_document(run_dir) or run_dir.name
    res_data = _read_json(run_dir / "04_result.json") or {}
    kv_results = res_data.get("kv_results") or []

    pred_by_key: Dict[str, Dict[str, Any]] = {}
    if isinstance(kv_results, list):
        for item in kv_results:
            if isinstance(item, dict) and item.get("key"):
                pred_by_key[str(item["key"])] = item

    eval_data = _read_json(run_dir / "05_eval.json")
    if not isinstance(eval_data, dict) or not eval_data.get("per_key"):
        computed = load_or_compute_run_eval(run_dir)
        if isinstance(computed, dict) and computed.get("per_key"):
            eval_data = computed
        elif not isinstance(eval_data, dict):
            eval_data = {}

    per_key = eval_data.get("per_key") or []
    eval_by_key: Dict[str, Dict[str, Any]] = {}
    if isinstance(per_key, list):
        for item in per_key:
            if isinstance(item, dict) and item.get("key"):
                eval_by_key[str(item["key"])] = item

    agentic_by_key = read_agentic_evals(run_dir)

    # Collect keys preserving original order
    ordered_keys: List[str] = []
    seen_keys = set()

    for item in kv_results:
        if isinstance(item, dict) and item.get("key"):
            k = str(item["key"])
            if k not in seen_keys:
                seen_keys.add(k)
                ordered_keys.append(k)

    for item in per_key:
        if isinstance(item, dict) and item.get("key"):
            k = str(item["key"])
            if k not in seen_keys:
                seen_keys.add(k)
                ordered_keys.append(k)

    for k in agentic_by_key:
        k_str = str(k)
        if k_str not in seen_keys:
            seen_keys.add(k_str)
            ordered_keys.append(k_str)

    rows: List[Dict[str, str]] = []
    for key in ordered_keys:
        p_item = pred_by_key.get(key, {})
        e_item = eval_by_key.get(key, {})
        a_item = agentic_by_key.get(key, {})

        # 1. Prediction
        pred = p_item.get("value")
        if pred is None:
            pred = e_item.get("value", {}).get("pred")
        pred_str = str(pred) if pred is not None else ""

        # 2. Ground Truth
        gt = e_item.get("value", {}).get("gold")
        gt_str = str(gt) if gt is not None else ""

        # 3. 채점결과
        is_correct = a_item.get("is_correct_answer")
        exact_match = e_item.get("value", {}).get("exact_match")
        status = a_item.get("status")
        verdict = _format_verdict(is_correct, exact_match, status)

        # 4. 검색페이지
        search_pages = p_item.get("search_pages")
        if not search_pages and isinstance(e_item.get("search_pages"), dict):
            search_pages = e_item["search_pages"].get("pred")
        pages_str = _format_search_pages(search_pages)

        # 5. 추출근거
        evidence = p_item.get("evidence_quote") or p_item.get("value_reason")
        if not evidence:
            raw_ev = p_item.get("evidence")
            if isinstance(raw_ev, list):
                evidence = "\n".join(
                    str(x.get("text", "")).strip()
                    for x in raw_ev
                    if isinstance(x, dict) and x.get("text")
                )
        evidence_str = str(evidence).strip() if evidence else ""

        # 6. 검색근거
        search_reasons = p_item.get("search_reasons")
        reasons_str = _format_search_reasons(search_reasons)

        # 7. 검토의견-요약
        reason_summary = a_item.get("reason_summary") or a_item.get("reason") or ""
        reason_summary_str = str(reason_summary).strip()

        # 8. 검토의견-상세
        reason_detail = a_item.get("reason_detail") or a_item.get("text") or ""
        reason_detail_str = str(reason_detail).strip()

        rows.append({
            "파일명": doc_name,
            "Key": key,
            "Prediction": pred_str,
            "Ground Truth": gt_str,
            "채점결과": verdict,
            "검색페이지": pages_str,
            "추출근거": evidence_str,
            "검색근거": reasons_str,
            "검토의견-요약": reason_summary_str,
            "검토의견-상세": reason_detail_str,
        })

    return rows


def _populate_detailed_sheet(
    ws: Any,
    run_dirs: Sequence[Path],
    *,
    header_font: Font,
    header_fill: PatternFill,
    header_align: Alignment,
    header_border: Border,
    data_font: Font,
    data_border: Border,
    correct_fill: PatternFill,
    incorrect_fill: PatternFill,
    evaluating_fill: PatternFill,
    center_align: Alignment,
    left_align_mid: Alignment,
    left_align_top: Alignment,
) -> None:
    ws.title = "채점결과"
    ws.append(EXCEL_COLUMNS)
    ws.row_dimensions[1].height = 28

    for col_idx in range(1, len(EXCEL_COLUMNS) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align
        cell.border = header_border

    row_idx = 2
    for rdir in run_dirs:
        rows = extract_run_rows(rdir)
        for rdata in rows:
            row_values = [rdata.get(col, "") for col in EXCEL_COLUMNS]
            ws.append(row_values)

            verdict = rdata.get("채점결과", "")
            for col_idx, col_name in enumerate(EXCEL_COLUMNS, start=1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.font = data_font
                cell.border = data_border

                if col_name in ("파일명", "Key", "채점결과", "검색페이지"):
                    cell.alignment = center_align
                elif col_name in ("Prediction", "Ground Truth"):
                    cell.alignment = left_align_mid
                else:  # 추출근거, 검색근거, 검토의견-요약, 검토의견-상세
                    cell.alignment = left_align_top

                if col_name == "채점결과":
                    if verdict in ("정답", "정답 (EM)"):
                        cell.fill = correct_fill
                        cell.font = Font(name="Malgun Gothic", size=10, bold=True, color="276A3C")
                    elif verdict in ("오답", "오답 (EM)"):
                        cell.fill = incorrect_fill
                        cell.font = Font(name="Malgun Gothic", size=10, bold=True, color="C00000")
                    elif verdict == "평가중":
                        cell.fill = evaluating_fill
                        cell.font = Font(name="Malgun Gothic", size=10, bold=True, color="B25E00")

            row_idx += 1

    ws.freeze_panes = "A2"
    column_widths = {
        "파일명": 32,
        "Key": 24,
        "Prediction": 20,
        "Ground Truth": 20,
        "채점결과": 12,
        "검색페이지": 12,
        "추출근거": 42,
        "검색근거": 42,
        "검토의견-요약": 36,
        "검토의견-상세": 55,
    }
    for col_idx, col_name in enumerate(EXCEL_COLUMNS, start=1):
        letter = get_column_letter(col_idx)
        ws.column_dimensions[letter].width = column_widths.get(col_name, 20)

    last_col_letter = get_column_letter(len(EXCEL_COLUMNS))
    total_rows = max(row_idx - 1, 1)
    ws.auto_filter.ref = f"A1:{last_col_letter}{total_rows}"


def _populate_per_run_summary_sheet(
    ws: Any,
    summary: Dict[str, Any],
    *,
    header_font: Font,
    header_fill: PatternFill,
    header_align: Alignment,
    header_border: Border,
    data_font: Font,
    data_border: Border,
    center_align: Alignment,
    left_align_mid: Alignment,
) -> None:
    ws.title = "Per-run summary"
    ws.append(PER_RUN_COLUMNS)
    ws.row_dimensions[1].height = 28

    for col_idx in range(1, len(PER_RUN_COLUMNS) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align
        cell.border = header_border

    per_run = summary.get("per_run") or []
    row_idx = 2

    for r in per_run:
        baseline = r.get("baseline") or {}
        agentic = r.get("agentic") or {}
        has_baseline = bool(r.get("has_baseline_eval"))

        em_val = baseline.get("value_exact_match") if has_baseline else None
        page_f1 = baseline.get("page_f1_macro") if has_baseline else None
        evid_f1 = baseline.get("evidence_token_f1") if has_baseline else None

        done_cnt = agentic.get("n_done", 0)
        tot_cnt = agentic.get("n_total", 0)
        done_str = f"{done_cnt}/{tot_cnt}" if (done_cnt or tot_cnt) else "0/0"
        pred_acc = agentic.get("accuracy")
        gt_valid = agentic.get("gold_validity")

        row_values = [
            r.get("run_id", ""),
            r.get("document", "") or r.get("run_id", ""),
            em_val if em_val is not None else "—",
            page_f1 if page_f1 is not None else "—",
            evid_f1 if evid_f1 is not None else "—",
            done_str,
            pred_acc if pred_acc is not None else "—",
            gt_valid if gt_valid is not None else "—",
        ]
        ws.append(row_values)
        ws.row_dimensions[row_idx].height = 22

        for col_idx, col_name in enumerate(PER_RUN_COLUMNS, start=1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.font = data_font
            cell.border = data_border
            if col_name == "파일명":
                cell.alignment = left_align_mid
            else:
                cell.alignment = center_align

            # Percentage formatting
            if col_name in ("Value EM", "Page F1", "Evid F1", "Pred Acc", "GT Valid"):
                if isinstance(cell.value, (int, float)):
                    cell.number_format = "0.0%"

        row_idx += 1

    # Average row when multiple runs are present
    if len(per_run) > 1:
        avg = summary.get("average") or {}
        avg_em = avg.get("value_exact_match")
        avg_page = avg.get("page_f1_macro")
        avg_evid = avg.get("evidence_token_f1")
        avg_done = avg.get("agentic_done_avg", 0)
        avg_tot = avg.get("agentic_total_avg", 0)
        tot_done = avg.get("agentic_done_total", 0)
        tot_tot = avg.get("agentic_total_total", 0)
        avg_done_str = f"{avg_done:.1f}/{avg_tot:.1f} ({tot_done}/{tot_tot})" if (tot_done or tot_tot) else "0/0"
        avg_acc = avg.get("accuracy")
        avg_gv = avg.get("gold_validity")

        avg_values = [
            "Average",
            f"{len(per_run)} runs",
            avg_em if avg_em is not None else "—",
            avg_page if avg_page is not None else "—",
            avg_evid if avg_evid is not None else "—",
            avg_done_str,
            avg_acc if avg_acc is not None else "—",
            avg_gv if avg_gv is not None else "—",
        ]
        ws.append(avg_values)
        ws.row_dimensions[row_idx].height = 24

        avg_font = Font(name="Malgun Gothic", size=10, bold=True, color="1F4E79")
        avg_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
        avg_border = Border(
            left=Side(style="thin", color="B0C4DE"),
            right=Side(style="thin", color="B0C4DE"),
            top=Side(style="thin", color="1F4E79"),
            bottom=Side(style="double", color="1F4E79"),
        )

        for col_idx, col_name in enumerate(PER_RUN_COLUMNS, start=1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.font = avg_font
            cell.fill = avg_fill
            cell.border = avg_border
            if col_name == "파일명":
                cell.alignment = left_align_mid
            else:
                cell.alignment = center_align

            if col_name in ("Value EM", "Page F1", "Evid F1", "Pred Acc", "GT Valid"):
                if isinstance(cell.value, (int, float)):
                    cell.number_format = "0.0%"

        row_idx += 1

    ws.freeze_panes = "A2"
    column_widths = {
        "Run ID": 42,
        "파일명": 38,
        "Value EM": 14,
        "Page F1": 14,
        "Evid F1": 14,
        "Agentic Done": 24,
        "Pred Acc": 14,
        "GT Valid": 14,
    }
    for col_idx, col_name in enumerate(PER_RUN_COLUMNS, start=1):
        letter = get_column_letter(col_idx)
        ws.column_dimensions[letter].width = column_widths.get(col_name, 16)

    last_col_letter = get_column_letter(len(PER_RUN_COLUMNS))
    filter_end_row = row_idx - 2 if len(per_run) > 1 else row_idx - 1
    if filter_end_row >= 2:
        ws.auto_filter.ref = f"A1:{last_col_letter}{filter_end_row}"


def _populate_key_x_run_matrix_sheet(
    ws: Any,
    summary: Dict[str, Any],
    *,
    header_font: Font,
    header_fill: PatternFill,
    header_align: Alignment,
    header_border: Border,
    data_font: Font,
    data_border: Border,
    correct_fill: PatternFill,
    incorrect_fill: PatternFill,
    evaluating_fill: PatternFill,
    center_align: Alignment,
    left_align_mid: Alignment,
    left_align_top: Alignment,
) -> None:
    ws.title = "Key x run matrix"

    per_run = summary.get("per_run") or []
    run_ids = [r.get("run_id") for r in per_run if r.get("run_id")]

    # Build header row: Key, Ground Truth, Overall (EM Rate), followed by runs
    matrix_headers = ["Key", "Ground Truth", "Overall (EM Rate)"]
    for r in per_run:
        doc = r.get("document") or r.get("run_id") or ""
        rid = r.get("run_id") or ""
        matrix_headers.append(f"{doc}\n({rid})")

    ws.append(matrix_headers)
    ws.row_dimensions[1].height = 36

    for col_idx in range(1, len(matrix_headers) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align
        cell.border = header_border

    per_key = summary.get("per_key") or []
    row_idx = 2

    for key_row in per_key:
        key = key_row.get("key", "")
        gold_val = str(key_row.get("gold_value") or "")
        ov = key_row.get("overall") or {}
        tot = ov.get("total", 0)
        corr = ov.get("correct", 0)
        rate = ov.get("rate")

        if tot > 0 and rate is not None:
            ov_text = f"{rate * 100:.1f}% ({corr}/{tot})"
        else:
            ov_text = "—"

        row_cells: List[str] = [key, gold_val, ov_text]

        by_run = key_row.get("by_run") or {}
        for rid in run_ids:
            cell_data = by_run.get(rid) or {}
            b_em = cell_data.get("baseline_em")
            ae = cell_data.get("agentic") or {}
            p_f1 = cell_data.get("page_f1")
            e_f1 = cell_data.get("evidence_f1")
            pred_val = cell_data.get("pred_value")

            em_label = "EM: Y" if b_em is True else ("EM: N" if b_em is False else "EM: —")
            ae_status = ae.get("status")
            if ae_status == "done":
                v = "정답" if ae.get("is_correct_answer") == "correct" else (
                    "오답" if ae.get("is_correct_answer") == "incorrect" else ae.get("is_correct_answer") or ""
                )
                gv = "GT: 유효" if ae.get("is_valid_gold") == "valid" else (
                    "GT: 무효" if ae.get("is_valid_gold") == "invalid" else ""
                )
                parts = [p for p in (v, gv) if p]
                ae_label = f"Agentic: {' · '.join(parts)}"
            elif ae_status == "running":
                ae_label = "Agentic: 평가중"
            elif ae_status == "error":
                ae_label = "Agentic: 오류"
            else:
                ae_label = ""

            l1 = f"[{em_label}] {ae_label}".strip()

            pred_str = str(pred_val).strip() if pred_val is not None else ""
            l2 = f"예측: {pred_str}" if pred_str else ""

            pf1_str = f"{p_f1 * 100:.1f}%" if p_f1 is not None else "—"
            ef1_str = f"{e_f1 * 100:.1f}%" if e_f1 is not None else "—"
            l3 = f"pageF1: {pf1_str} · evidF1: {ef1_str}"

            lines = [l for l in (l1, l2, l3) if l]
            cell_text = "\n".join(lines) if lines else "—"
            row_cells.append(cell_text)

        ws.append(row_cells)
        ws.row_dimensions[row_idx].height = 48

        # Style cells
        # Col 1: Key
        c1 = ws.cell(row=row_idx, column=1)
        c1.font = data_font
        c1.alignment = left_align_mid
        c1.border = data_border

        # Col 2: Ground Truth
        c2 = ws.cell(row=row_idx, column=2)
        c2.font = data_font
        c2.alignment = left_align_mid
        c2.border = data_border

        # Col 3: Overall
        c3 = ws.cell(row=row_idx, column=3)
        c3.font = Font(name="Malgun Gothic", size=10, bold=True)
        c3.alignment = center_align
        c3.border = data_border
        if rate == 1.0:
            c3.fill = correct_fill
            c3.font = Font(name="Malgun Gothic", size=10, bold=True, color="276A3C")
        elif rate == 0.0 and tot > 0:
            c3.fill = incorrect_fill
            c3.font = Font(name="Malgun Gothic", size=10, bold=True, color="C00000")
        elif rate is not None and 0.0 < rate < 1.0:
            c3.fill = evaluating_fill
            c3.font = Font(name="Malgun Gothic", size=10, bold=True, color="B25E00")

        # Col 4..N: Runs
        for idx, rid in enumerate(run_ids, start=4):
            cell = ws.cell(row=row_idx, column=idx)
            cell.font = Font(name="Malgun Gothic", size=9.5)
            cell.alignment = left_align_top
            cell.border = data_border

            cell_data = by_run.get(rid) or {}
            b_em = cell_data.get("baseline_em")
            ae = cell_data.get("agentic") or {}
            ae_correct = ae.get("is_correct_answer")
            ae_status = ae.get("status")

            if ae_correct == "correct":
                cell.fill = correct_fill
            elif ae_correct == "incorrect":
                cell.fill = incorrect_fill
            elif b_em is True:
                cell.fill = correct_fill
            elif b_em is False:
                cell.fill = incorrect_fill
            elif ae_status == "running":
                cell.fill = evaluating_fill

        row_idx += 1

    # Freeze Key, Ground Truth, and Overall columns
    ws.freeze_panes = "D2"

    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 22
    ws.column_dimensions["C"].width = 18
    for idx in range(4, len(matrix_headers) + 1):
        letter = get_column_letter(idx)
        ws.column_dimensions[letter].width = 34

    last_col_letter = get_column_letter(len(matrix_headers))
    total_rows = max(row_idx - 1, 1)
    ws.auto_filter.ref = f"A1:{last_col_letter}{total_rows}"


def generate_evaluation_xlsx(
    run_dirs: Sequence[Path],
    runs_root: Optional[Path] = None,
) -> io.BytesIO:
    """Generate Excel workbook containing 채점결과, Per-run summary, and Key x run matrix sheets."""
    wb = openpyxl.Workbook()

    # Common styling objects
    header_font = Font(name="Malgun Gothic", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

    thin_border_side = Side(style="thin", color="D3D3D3")
    data_border = Border(
        left=thin_border_side,
        right=thin_border_side,
        top=thin_border_side,
        bottom=thin_border_side,
    )
    header_border = Border(
        left=Side(style="thin", color="0D233A"),
        right=Side(style="thin", color="0D233A"),
        top=Side(style="thin", color="0D233A"),
        bottom=Side(style="medium", color="0D233A"),
    )

    correct_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")  # soft green
    incorrect_fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")  # soft red
    evaluating_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")  # soft yellow

    center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left_align_mid = Alignment(horizontal="left", vertical="center", wrap_text=True)
    left_align_top = Alignment(horizontal="left", vertical="top", wrap_text=True)
    data_font = Font(name="Malgun Gothic", size=10)

    # Sheet 1: 채점결과
    ws_detail = wb.active
    _populate_detailed_sheet(
        ws_detail,
        run_dirs,
        header_font=header_font,
        header_fill=header_fill,
        header_align=header_align,
        header_border=header_border,
        data_font=data_font,
        data_border=data_border,
        correct_fill=correct_fill,
        incorrect_fill=incorrect_fill,
        evaluating_fill=evaluating_fill,
        center_align=center_align,
        left_align_mid=left_align_mid,
        left_align_top=left_align_top,
    )

    # Derive evaluation summary for sheets 2 and 3
    if runs_root is None:
        runs_root = run_dirs[0].parent if run_dirs else Path(".")
    run_ids = [r.name for r in run_dirs]

    summary: Dict[str, Any] = {}
    try:
        summary = build_evaluation_summary(run_ids, runs_root)
    except Exception:
        pass

    # Sheet 2: Per-run summary
    ws_summary = wb.create_sheet(title="Per-run summary")
    _populate_per_run_summary_sheet(
        ws_summary,
        summary,
        header_font=header_font,
        header_fill=header_fill,
        header_align=header_align,
        header_border=header_border,
        data_font=data_font,
        data_border=data_border,
        center_align=center_align,
        left_align_mid=left_align_mid,
    )

    # Sheet 3: Key x run matrix
    ws_matrix = wb.create_sheet(title="Key x run matrix")
    _populate_key_x_run_matrix_sheet(
        ws_matrix,
        summary,
        header_font=header_font,
        header_fill=header_fill,
        header_align=header_align,
        header_border=header_border,
        data_font=data_font,
        data_border=data_border,
        correct_fill=correct_fill,
        incorrect_fill=incorrect_fill,
        evaluating_fill=evaluating_fill,
        center_align=center_align,
        left_align_mid=left_align_mid,
        left_align_top=left_align_top,
    )

    # Ensure sheet 1 is the active view upon opening
    wb.active = ws_detail

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf
