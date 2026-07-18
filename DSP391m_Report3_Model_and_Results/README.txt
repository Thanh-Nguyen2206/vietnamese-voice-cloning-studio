====================================================================
DSP391m — Data Science Capstone Project
Report 3: Model & Results
(Model Development + Model Evaluation and Fine-Tuning +
 Results Interpretation and Visualization + Conclusion and Recommendations)

Project: Vietnamese Voice Cloning Studio — objective benchmarking of
         zero-shot Text-to-Speech with F5-TTS ViVoice.

Team Members:
  - Nguyễn Hoàng Thanh - SE172535
  - Trương Thanh Tuấn  - SE182217
  - Nguyễn Thị Vân Anh  - DE18037
Instructor:      Nguyễn Quốc Trung
Academic Term:   Summer 2026
====================================================================

WHAT TO READ FIRST
------------------
  DSP391m_Report3_Model_and_Results.pdf   <- the report (grade this)
  DSP391m_Report3_Model_and_Results.docx  <- same report, editable Word source

The report follows the official capstone template and covers, in order:
  1  Introduction and Background        7  Model Development (architecture, training)
  2  Literature Review                  8  Model Evaluation and Fine-Tuning
  3  Data Description                    9  Results Interpretation and Visualization
  4  Data Cleaning and Preprocessing   10  Conclusion and Recommendations (Provisional)
  5  Exploratory Data Analysis         11  License and Ethical Considerations
  6  Methodology                            References + Appendices

FOLDER CONTENTS
---------------
  results/            Raw experimental evidence behind Section 9.
    results.json      Per-sentence + aggregate metrics for all 6 engines (machine-readable).
    results.csv       Same data as a spreadsheet.
    report.md         Auto-generated summary table (mean/median WER, CER, SECS, RTF).
    manifest.json     The evaluation manifest (which audio, text, seed, NFE per case).
    figures/          The 5 charts used in the report (WER/CER, speaker similarity,
                      speed, quality trade-off, per-sentence WER distribution).

  audio_samples/      The ACTUAL generated audio (42 files = 6 engines x 7 sentences).
                      Naming: <engine>__<sentence-id>.wav. This is the real model output
                      that was measured — provided as direct evidence, not a claim.

  code/               The code that produced the results (evidence of methodology).
    run_benchmark.py      Generates audio for all engines over the test sentences.
    evaluate.py           Computes WER/CER (Whisper) + speaker similarity (Resemblyzer) + RTF.
    plot_results.py       Renders the figures from results.json.
    build_report_docx.py  Builds the .docx report (fully data-driven from results.json).
    train.py              Fine-tuning pipeline (Model Development / Training Procedure).
    data_prep.py          Audio pre-processing / dataset preparation.
    compare_models.py     CLI multi-engine comparison used during development.
    app.py, engines.py    The inference application and the 5 comparison engines.
    configs/train_config.yaml   DiT architecture + training hyper-parameters.
    voice_studio/         Offline-testable core (evaluation, audio metrics, text processing).

EXPERIMENT SUMMARY (n = 7 sentences per engine, CPU, seed 42, NFE 32)
---------------------------------------------------------------------
  Metrics: WER/CER via Whisper ASR round-trip; SECS = cosine similarity of
  Resemblyzer speaker embeddings; RTF = inference time / audio duration.

  Engine     WER% (median/mean)   SECS    RTF
  F5-TTS         8.7 / 22.4       0.862   22.8   <- recommended primary (VI pre-trained)
  XTTS-v2       11.1 / 21.8       0.877    2.9   <- equally strong, faster on CPU
  Edge-TTS       9.1 / 23.9       0.565    0.55
  Piper          9.1 / 27.1       0.637    0.15
  MMS-TTS       36.4 / 43.2       0.618    0.39
  Bark         125.0 / 145.3      0.469    7.48  <- no Vietnamese support (lower bound)

  Key finding: only the two voice-cloning engines (F5-TTS, XTTS-v2) exceed the
  0.75 speaker-similarity threshold; the fixed-voice engines render a default
  voice, not the reference speaker. Conclusions are PROVISIONAL: they describe
  the zero-shot base model over 7 sentences, not a fine-tuned or large-scale run.

HOW TO REPRODUCE (requires the project environment with the TTS stack)
----------------------------------------------------------------------
  python code/run_benchmark.py
  python code/evaluate.py     --manifest outputs/benchmark/manifest.json \
                              --output-dir outputs/benchmark/evaluation
  python code/plot_results.py --results outputs/benchmark/evaluation/results.json \
                              --output-dir outputs/benchmark/figures_en --lang en
  python code/build_report_docx.py

NOTE
----
  Speaker similarity (SECS) is a relative quality metric, NOT biometric identity
  verification. Voice cloning must only be used with the speaker's consent.
