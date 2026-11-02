# Data-use terms

What each source allows and what it means for this project. Read on 2 November 2026 (RSNA Kaggle rules); the Zenodo licence and RSNA challenge terms were read on 30 October 2026. This is the team's reading, not legal advice.

## ISLES'24 (Zenodo record 17652035)

- Licence: CC BY-NC-SA 4.0. Attribution, non-commercial use, and derivatives under the same licence.
- The archive and the slim tarball can sit on the shared Drive for the team. The slim tarball stays under the same licence.
- Images can appear in the report and demo, with attribution.
- Cite the dataset paper (arXiv:2408.11142) and the challenge paper (arXiv:2408.10966), plus TopCoW if the Circle of Willis masks are used.

## RSNA Intracranial Haemorrhage (Kaggle copy)

We use the Kaggle copy, so the Kaggle competition rules apply. They are stricter than the RSNA challenge Terms of Use and they match the AWS/MIRA listing on redistribution.

- **Allowed:** non-commercial use, including academic research and education.
- **Not allowed:** redistributing the data, giving it to anyone who is not a competition participant, or trying to re-identify anyone. Identifying information found in the data must be reported on the competition forum.
- **Copies:** the rules also say participants agree not to make copies. See the open question below.
- **Attribution:** link the dataset and cite Flanders et al., *Radiol Artif Intell* 2020;2:3.

What this means for us:

| Item | Rule |
|---|---|
| Teammates | Each must accept the competition rules on Kaggle before getting access to the `rsna-slim-v1-*` datasets |
| `rsna-slim-v1-*` datasets and notebooks using them | Stay private on Kaggle |
| Google Drive | No RSNA images, shards or indexes |
| This repo | No RSNA images, indexes, or slice, patient or study IDs |
| Local copies | Delete downloaded shards once the datasets are checked |
| Report and demo | No RSNA images; use ISLES'24 images instead. Aggregate metrics are fine |
| Model weights | The rules do not mention them. Keep them private to the team and do not publish them |

The competition's limit on sharing code privately applied only during the competition period, which ended in 2019.

**Open question for the professors:** taken literally, the no-copies clause conflicts with storing preprocessed shards, even privately (DEC-17). Our position is that private shards on the same platform, available only to people who accepted the rules, for academic work, follow the purpose of the rules. It has not been confirmed. State it in the report's data section.

## Guideline documents

- 2026 AHA/ASA guideline, DAWN and DEFUSE-3 papers: licence terms not checked yet. Treat them as copyrighted until then.
- PDFs stay in `StrokeCopilot/artifacts/guidelines/` on Drive, not in the repo.
- Check the terms before ingestion in Phase 2, because `docs/RULEBOOK.md` and the corpus would hold passages from them.

## Patient data and the demo

Both imaging datasets are de-identified. Every demo screen carries a visible "research prototype, not for clinical use" label.
