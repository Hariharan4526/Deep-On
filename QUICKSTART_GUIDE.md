# Deep-On Enhancements - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Review the Enhancements
```bash
# Read the complete integration guide
cat ENHANCEMENT_GUIDE.md

# Read the implementation summary
cat IMPLEMENTATION_SUMMARY.md
```

### Step 2: Update Your Configuration
Edit `config.yaml` to enable the enhancements you want:

**Minimal Enhancement (Fast)**:
```yaml
detection:
  adaptive_frame_processing: true

temporal:
  enabled: false  # Skip for speed

forensic:
  enabled: false  # Skip for speed

robustness:
  enabled: false
```

**Full Enhancement (Best Accuracy)**:
```yaml
detection:
  adaptive_frame_processing: true

temporal:
  enabled: true
  use_lstm: true

forensic:
  enabled: true

robustness:
  enabled: true
  test_time_augmentation: false  # Too slow for real-time
```

**Balanced Enhancement (Recommended)**:
```yaml
detection:
  adaptive_frame_processing: true

temporal:
  enabled: true
  use_lstm: true

forensic:
  enabled: true

robustness:
  enabled: false
```

### Step 3: Test It
```bash
# Test on a single video
python main.py detect --video test_video.mp4

# Test on a directory
python main.py detect --dir videos/ --output results/

# Evaluate on a CSV
python main.py evaluate --csv labels.csv
```

---

## 📊 Expected Improvements by Issue

### Issue: High-Quality Deepfakes
```yaml
# Recommended config
temporal:
  enabled: true
  use_lstm: true

forensic:
  enabled: true
```
**Why**: Temporal analysis detects unnatural motion; forensic detects subtle artifacts

### Issue: Compressed Videos
```yaml
# Recommended config
detection:
  adaptive_frame_processing: true

robustness:
  enabled: true
  training_augmentation: true
```
**Why**: Adaptive processor skips corrupted frames; robustness trains on compression artifacts

### Issue: Low-Light Videos
```yaml
# Recommended config
detection:
  adaptive_frame_processing: true
  # Adjust blur threshold for low light
  blur_threshold: 40.0  # Lower = more lenient
```
**Why**: Adaptive processor normalizes brightness automatically

### Issue: Side-Face Angles
```yaml
# Recommended config
detection:
  adaptive_frame_processing: true

robustness:
  enabled: true
  training_augmentation: true
```
**Why**: Adaptive quality filtering + rotation invariance training

### Issue: Fast Motion Videos
```yaml
# Recommended config
detection:
  adaptive_frame_processing: true
  motion_threshold: 0.03  # Lower = more motion-sensitive

temporal:
  enabled: true
```
**Why**: Motion-sensitive extraction + temporal pattern analysis

### Issue: AI-Generated Talking-Face
```yaml
# Recommended config
temporal:
  enabled: true
  use_lstm: true
```
**Why**: Temporal analyzer detects lip-sync inconsistencies and unnatural blinking

---

## 🔍 Monitoring & Tuning

### Check If Enhancements Are Working

**Enable Verbose Logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

from src.inference import DeepfakeDetector
detector = DeepfakeDetector()
```

**Monitor Frame Selection** (with adaptive frame processor):
- If too many frames skipped: Lower `blur_threshold` or `duplicate_threshold`
- If including blurry frames: Raise `blur_threshold`

**Monitor Temporal Analysis**:
- Check if `temporal_analyzer.use_lstm` is True (or falls back to False if torch unavailable)
- Review flickering & motion anomaly scores

**Monitor Forensic Analysis**:
- Higher forensic scores indicate more GAN/artifact patterns
- Tune sensitivity if too many false positives

### Performance Tuning

**For Real-Time Inference** (prioritize speed):
```yaml
detection:
  adaptive_frame_processing: false
  frame_skip: 5  # Keep original

temporal:
  enabled: false

forensic:
  enabled: false

robustness:
  test_time_augmentation: false
```

**For Batch Evaluation** (prioritize accuracy):
```yaml
detection:
  adaptive_frame_processing: true

temporal:
  enabled: true

forensic:
  enabled: true

robustness:
  enabled: true
  test_time_augmentation: true
  tta_num_augmentations: 4
```

**For Balanced Performance**:
```yaml
detection:
  adaptive_frame_processing: true

temporal:
  enabled: true
  use_lstm: false  # Use statistics, faster

forensic:
  enabled: true

robustness:
  enabled: false  # Skip TTA
```

---

## 📈 Evaluation with New Metrics

### Generate Comprehensive Evaluation Report

Create `evaluate_enhanced.py`:
```python
#!/usr/bin/env python3
"""Enhanced evaluation using new metrics."""
import csv
from src.inference import DeepfakeDetector
from src.evaluation_metrics import EvaluationMetrics
from src.utils import save_json

# Load detector
detector = DeepfakeDetector()
evaluator = EvaluationMetrics()

# Read test videos
y_true, y_pred, y_probs = [], [], []
with open("test_videos.csv") as f:
    for row in csv.DictReader(f):
        video_path = row["path"]
        ground_truth = int(row["label"])  # 0=real, 1=fake
        
        result = detector.process_video(video_path)
        pred = 1 if result["classification"] == "Deepfake" else 0
        prob = result["probability"]
        
        y_true.append(ground_truth)
        y_pred.append(pred)
        y_probs.append(prob)

# Comprehensive evaluation
metrics = evaluator.evaluate_comprehensive(y_true, y_pred, y_probs)

# Save report
save_json(metrics, "evaluation_report.json")

# Print summary
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"F1-Score: {metrics['f1']:.4f}")
print(f"Confusion Matrix: {metrics['confusion_matrix']}")
```

**Run it**:
```bash
python evaluate_enhanced.py
```

---

## 🐛 Troubleshooting

### Enhancements Not Loading
```bash
# Check if modules can be imported
python -c "from src.adaptive_frame_processor import AdaptiveFrameProcessor"
python -c "from src.temporal_analyzer import TemporalAnalyzer"
python -c "from src.forensic_analyzer import ForensicAnalyzer"
python -c "from src.robustness_augmentation import RobustnessAugmentation"
python -c "from src.evaluation_metrics import EvaluationMetrics"
```

If any fails, check that the files exist in `src/`.

### LSTM Not Available
```python
from src.temporal_analyzer import TemporalAnalyzer
analyzer = TemporalAnalyzer(use_lstm=True)
print(f"LSTM available: {analyzer.use_lstm}")  # Should be True
```

If False, LSTM falls back to statistical analysis (still useful but slightly less accurate).

### Slow Inference
**Disable slow features**:
```yaml
robustness:
  test_time_augmentation: false  # This is the main culprit

forensic:
  enabled: false  # Skip if speed critical
```

### Memory Issues
```yaml
hardware:
  batch_size: 4  # Reduce from 8
  num_workers: 2  # Reduce from 4
  max_batch_size: 16  # Reduce from 32
```

---

## 📚 Deep Dive

### For More Details, See:

1. **Complete Integration Guide**: `ENHANCEMENT_GUIDE.md`
   - Architecture overview
   - Detailed configuration reference
   - Advanced usage patterns
   - Performance considerations

2. **Implementation Summary**: `IMPLEMENTATION_SUMMARY.md`
   - What was implemented
   - File structure
   - Backward compatibility notes
   - Performance metrics

3. **Module Documentation**:
   - `src/adaptive_frame_processor.py` — Frame selection
   - `src/temporal_analyzer.py` — Motion analysis
   - `src/forensic_analyzer.py` — Artifact detection
   - `src/robustness_augmentation.py` — Augmentation
   - `src/evaluation_metrics.py` — Metrics

---

## ✅ Checklist for Testing

- [ ] Read ENHANCEMENT_GUIDE.md
- [ ] Read IMPLEMENTATION_SUMMARY.md
- [ ] Verify new modules can be imported
- [ ] Update config.yaml with desired enhancements
- [ ] Test on 1-2 problem videos
- [ ] Compare results with/without enhancements
- [ ] Run comprehensive evaluation
- [ ] Tune configuration based on results
- [ ] Update train.py if doing transfer learning
- [ ] Monitor performance metrics

---

## 🎯 Recommended First Steps

1. **Start with adaptive frame processing** (lowest risk, fast):
   ```yaml
   detection:
     adaptive_frame_processing: true
   ```
   Test on compressed/low-light videos.

2. **Add temporal analysis** (medium risk, good improvement):
   ```yaml
   temporal:
     enabled: true
     use_lstm: true
   ```
   Test on AI-generated talking-face videos.

3. **Add forensic analysis** (low cost, good signal):
   ```yaml
   forensic:
     enabled: true
   ```
   Test on high-quality deepfakes.

4. **Only if needed, add augmentation/TTA** (high cost):
   ```yaml
   robustness:
     enabled: true
     test_time_augmentation: true
     tta_num_augmentations: 4
   ```
   Use for batch evaluation only, not real-time.

---

## 💡 Pro Tips

1. **Always test on representative videos** before enabling all enhancements
2. **Monitor false positive rates** — tune thresholds if needed
3. **Disable expensive features for real-time** (TTA, forensic if speed critical)
4. **Use ensemble voting** with test-time augmentation for highest accuracy
5. **Enable early stopping in training** to prevent overfitting
6. **Use class balancing** when training on mixed datasets

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the detailed guides (ENHANCEMENT_GUIDE.md)
3. Check module docstrings for API reference
4. Verify config.yaml syntax (YAML indentation matters)

Good luck! 🚀
