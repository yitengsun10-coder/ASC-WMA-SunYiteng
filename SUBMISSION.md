# Submission

- Student: 孙逸腾（240810010427）
- Task: Embodied World Model / WMA
- Final formal version: `final_amp20_v4`; 20/20 newly generated cases satisfy PSNR >= 25 dB and all manifest video specifications.
- AMP baseline: 7830.488590281457 s; precise inference total: 6137.218386184424 s.
- Exact speedup: 1.2759018984738715x; time reduction: 21.624068339727575%.
- Mean PSNR: 38.43932329089311 dB; minimum: 25.82416722996686 dB.
- Original wrapper exit 1 after inference because ffprobe was absent from PATH; CPU recovery audit exit 0. Both original records are retained.
- Final report: `final_report/`; no old-run video splicing or generation retry.
- Formal sending/receipt is not confirmed by repository publication.

## Legacy results retained for traceability

- Legacy quality status: 20/20 cases executed; 20/20 satisfy PSNR ≥ 25 dB
- 2026-09-28 performance status: pending a fresh, unified FP16/AMP baseline-versus-optimized run over all 20 cases. The archived single-case 1.085× result does not satisfy the new ≥1.25× total-speedup threshold.
- Average PSNR: 40.561815 dB
- Minimum PSNR: 25.0749163 dB (`unitree_z1_dual_arm_stackbox_v2/case1`)
- Audit trail: the original 24.8489438 dB result and failed seed controls remain archived; the final controlled eta schedule and hashes are stored under `official_20_cases_20260814/wma_case1_evidence/switch3_late090/`


