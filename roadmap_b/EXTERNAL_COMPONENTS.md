# External and generated components

- MELD annotations: declare-lab/MELD, revision e8cedf27b5d2877e198332c957127e16eb214afe.
  Source: https://github.com/declare-lab/MELD
  Public audit: evidence/data_audit/; exact annotation identities: baseline_lock.json.
  Media are user supplied and are not redistributed here. No project-wide software license is being newly selected in this commit; publication is not a grant
  of rights to redistribute the Friends television material.
- Text weights/tokenizer: FacebookAI/roberta-base, revision e2da8e2f811d1448a5b465c236feacd80ffbac7b.
  Model card: https://huggingface.co/FacebookAI/roberta-base (MIT).
- Audio weights: microsoft/wavlm-base-plus, revision 4c66d4806a428f2e922ccfa1a962776e232d487b.
  Model card: https://huggingface.co/microsoft/wavlm-base-plus (MIT).
  Its official preprocessing configuration disables waveform amplitude normalization.
- PyTorch and Transformers supply model implementations, autograd and checkpoint serialization.
  scikit-learn supplies classification metrics; SciPy supplies resampling and bounded temperature optimization.
  Exact package versions are in requirements.lock.txt and each run's environment.json.
- FFmpeg supplies decoding, channel conversion and 16-kHz conversion. When system FFmpeg is unavailable,
  imageio-ffmpeg supplies a bundled executable. Its binary licensing/build notices remain applicable.
- uv supplies environment management and is installed from its official installer if missing.

AI-assisted original components: causal packing/masks, preprocessing/audits, training orchestration,
fusion/calibration, replay/session API, authored response phrases, reports, smoke fixtures and tests.
No EMART code or external emotion classifier is copied into this implementation.

Generated fixtures use tones/noise and synthetic transcripts with arbitrary labels.
They are explicitly marked fixture=true and cannot support MELD performance or response-quality claims.
Pinned pretrained encoders are reused; final task-specific classifiers require training on official train.

Project integration, tests and analysis were developed with AI assistance. The project author is responsible for understanding, verifying and modifying them; tool-generated output is not independent validation. The bounded residual, matched controls, canonical numerical policy, runtime/session, authored responses and recording/listener tools are project work, not functionality supplied by pretrained emotion classifiers. Human recordings are real consenting-speaker inputs, never synthetic evidence; raw recordings and private listener information are excluded from Git. No pretrained weights are redistributed here. Third-party licenses/terms remain applicable; consult the linked original sources for their full notices.


## T+A closure additions

- CREMA-D: CheyneyComputerScience/CREMA-D, pinned revision `1658cd342dff90010aa843eaeebd53610a08b1dc`. D2 was trained on approved CREMA training partitions using aggregate audio-only listener votes in its distinct six-class namespace. The locally retained source notice specifies Open Database License1.0 for the database and Database Contents License1.0 for individual contents; these notices are not a blanket license for this project or its derived weights. Source and notices: https://github.com/CheyneyComputerScience/CREMA-D/tree/1658cd342dff90010aa843eaeebd53610a08b1dc . No registration form or agreement was submitted in this continuation.
- RAVDESS: existing acted-speech working predictions and published listener annotations supplied a conditional evaluation target. Retained provenance identifies the [RAVDESS media](https://zenodo.org/records/1188976), [dataset paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391), and [S5 raw-validation workbook](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0196391.s008&type=supplementary), workbook SHA256 `b304b75a1a1a8928be557b0fe6a248c40db193e95e4d5b02073dce8dd034fcee`. The recorded article/validation-data notice is CC BY. The media's recorded CC BY-NC-SA4.0 terms and annotation/article notices remain applicable. No media, individual listener records or per-record latent vectors are newly redistributed in this commit. Former confirmation has already been inspected; it is not a fresh holdout.
- D2 layer-mixture/summaries/head, portable asset verification, temporal wrapper, fixed H1 authored offers, evaluation and handoff documentation are AI-assisted project work. They do not constitute independent judgments of human empathy, emotion or response usefulness.

The selected B/D2 project weights remain external because redistribution permission is not newly established here. Exact pretrained revisions and project checkpoint hashes are in `evidence/ta_system_closure/ASSET_MANIFEST.json`. The33-file local bundle was verified and loaded; no asset download endpoint or public model backup is claimed. Keep original third-party notices with any separately authorized redistribution.
