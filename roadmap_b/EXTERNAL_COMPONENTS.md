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
