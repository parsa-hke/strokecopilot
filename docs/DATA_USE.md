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

Checked on 4 November 2026.

| Document | Terms | What this means |
|---|---|---|
| 2026 AHA/ASA guideline (*Stroke*) | © American Heart Association. Not to be reproduced "in whole or in part" without written permission ([permission guidelines](https://www.heart.org/en/about-us/statements-and-policies/copyright-permission-guidelines)). Educational permission covers at most 10% of the text, with a fee per item ([guideline material](https://www.heart.org/en/about-us/statements-and-policies/guidelines-for-use-of-guidelines-and-statements-material)) | No copied passages |
| DAWN (NEJM 2018), appendix and protocol | NEJM reserves all rights, including text and data mining and AI training, and bars derivative works made with AI tools, except personal noncommercial use ([NEJM copyright](https://www.nejmgroup.org/legal/copyright-information.htm)) | Not sent to any AI tool; no copied passages |
| DEFUSE-3 (NEJM 2018) | NEJM terms as above. The PMC author manuscript ([PMC6590673](https://pmc.ncbi.nlm.nih.gov/articles/PMC6590673/)) may be read and text mined under the [PMC copyright notice](https://pmc.ncbi.nlm.nih.gov/about/copyright/) | The PMC manuscript may be used to draft passages |

What we do (DEC-18):
- The corpus is `corpus/RULEBOOK.md`, a rulebook in our own words: criteria and numbers as facts, with a pointer to the section, recommendation and page. It holds no copied wording, so it can live in the repo and appear in the report.
- The PDFs stay in `StrokeCopilot/artifacts/guidelines/` on Drive, never in the repo, and are never sent to an LLM. People use them to check passages.
- Before a run, the copy check compares each passage with the PDF text locally and lists passages that share 12 or more consecutive words.
- Cite the guideline and both trial papers in the report.

**Open question for the professors:** whether an own-words rulebook derived from the guideline is enough, or whether they expect permission from AHA and NEJM. Ask it with the RSNA no-copies question.

## Patient data and the demo

Both imaging datasets are de-identified. Every demo screen carries a visible "research prototype, not for clinical use" label.
