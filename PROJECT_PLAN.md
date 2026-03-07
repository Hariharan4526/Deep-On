# Deepfake & AI Content Detection - MVP Plan

## What is implemented now
- `main.py`: CLI entrypoint for running analysis and exporting reports.
- `detect.py`: Detection engine with:
  - Media-type inference (`image`, `video`, `audio`, `unknown`)
  - Metadata extraction
  - Image artifact analysis (edge, sharpness, frequency signature)
  - Video sampling + temporal inconsistency analysis
  - Audio heuristic analysis (`.wav` waveform and spectral checks)
  - Confidence scoring and prototype source estimation

## Run instructions
1. Activate virtual environment:
   - PowerShell: `./env/Scripts/Activate.ps1`
2. Install required packages (if not already installed):
   - `pip install -r Requirements.txt`
3. Run detector:
   - `python main.py <path_to_media>`
4. Optional JSON output:
   - `python main.py <path_to_media> --json`
5. Save report to file:
   - `python main.py <path_to_media> --save reports/result.json`

## Suggested folder structure next
- `data/` for sample media
- `reports/` for detection reports
- `models/` for trained model weights (future)
- `notebooks/` for experiments (future)

## Next research upgrades
1. Dataset integration: FaceForensics++, DFDC, ASVspoof, FakeAVCeleb.
2. Train baseline deep models (Xception/EfficientNet + audio CNN).
3. Add explainability (Grad-CAM, saliency) and confidence calibration.
4. Build FastAPI endpoint for near real-time analysis.
5. Build dashboard for investigator workflow.

## Important note
Current source-model identification is heuristic and intended for research prototyping only, not legal-grade attribution.
