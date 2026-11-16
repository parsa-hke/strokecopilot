# Labelling guide for Hana (labeller B)

You are one of two people checking the **verifier**, the part of our system that decides whether each statement a
model writes is supported by its inputs and the rulebook. You are not being tested; the verifier is. Your labels tell
us how often it is right, both when it flags a claim and when it lets one pass. This is the human check the plan asks
for in Phase 3.

Four of us share it in two pairs: **Lorna with Benny, and Mohamed with Hana**. Your pair labels the same 50 claims,
each of you on your own, so every claim gets two independent readings. From those we measure how often you and
Mohamed agree (Cohen's kappa) and how often the verifier agrees with the two of you.

**In short:** 50 claims, each labelled **Error** or **No error**, on your own, in a webpage that runs in your browser.
About 3 minutes a claim, so roughly 2.5 hours in all: spread it over two or three sittings.

The full rules are in `docs/LABELLING_GUIDE.md`; the page has a summary under **Rules**.

## 1. Open your page

1. In our shared Google Drive folder, open **StrokeCopilot → labelling** and download **`labelling_hana.html`**.
   Use only this file: it is yours, and it saves your labels under your name.
2. Double-click the downloaded file. It opens in your browser (Chrome, Edge, Firefox or Safari). Do not open it inside
   Google Drive's preview, which shows the code instead, and do not use a private or incognito window.
3. The top bar should say **Hana · pair 2 with Mohamed · labeller B**. If a red box says the browser is not saving your labels, you
   are in a private window or storage is switched off: open it in a normal window.

Nothing is sent anywhere. The page works offline and keeps your labels in this browser on this computer.

## 2. What the page shows

- **Left:** your 50 claims. Their IDs skip numbers because the other pair has the rest; the top of
  the page also counts your own (for example "S0xx · 1 of 50"). A grey circle is still to do, green is No error, red is Error. **To do**
  shows only the ones left.
- **Top of the main area:** the claim you are judging, in large type, with where it came from (a reasoning step,
  a blocking factor, an uncertainty, or a decision's citation) and the case (C001, ...). Under it, **From the answer**
  shows the whole sentence the claim was taken from, for context.
- **Sources the model declared:** what the model said the claim rests on:
  - an **imaging field** with its value (that row is also highlighted in the imaging table);
  - a **note quote** (highlighted in the note). If it says *not found word for word in the note*, the model changed
    or shortened the note's words;
  - a **rulebook passage** with its full text. If it says *no passage with this ID exists*, the model cited a passage
    that is not in the rulebook.
- **Imaging findings** and **Triage note:** exactly what the model was given for this case. `null` means unknown.
- **Rulebook** (top bar): every passage the models could cite, with a search box.

The page never shows which model wrote the claim or what the verifier decided. That is deliberate: please do not try
to guess either.

## 3. How to decide

For each claim, ask two questions:

1. **Is the content true**, given the imaging findings, the note and the rulebook?
2. **Does the source the model cited support it?**

Choose **Error** if either answer is no. Choose **No error** if both are yes, or if the claim only restates the
answer's own decision ("thrombolysis is not recommended").

The rules that settle most cases:

- **Correct medicine is not enough.** A true clinical rule that no rulebook passage states is an error.
- **`null` means unknown.** Saying a null finding is unknown or not reported is fine; any value for it is an error.
- **Numbers** are compared after rounding the true value to the claim's precision: "9 h" matches 9 h 0 min, but
  "4 h" does not match 4 h 40 min.
- **Uncertainties** have no sources in the answer's format: judge only whether their content is true.
- **Blocking factors** and **decision citations** ("Passage X is a rule about thrombolysis") have no sources of their
  own: they may rest on the passages either decision cites. For a citation, check that the passage exists and is about
  that treatment.
- Whether an error would change the decision does not matter here; that is measured separately.

When you are unsure, choose the label these rules give and **write why in the note box**. Notes are the most useful
part when the two of you disagree.

Three examples from an earlier run:

| Claim | Label | Why |
|---|---|---|
| "BP 168/92 does not need lowering before thrombolysis" | No error | The rule says lower only if BP is 185/110 or more; 168/92 is below. |
| "This patient is eligible for thrombectomy", citing thrombolysis passages | Error | True, but the cited passages do not support it. |
| "Treatment cannot start before imaging" | Error | The rulebook says the opposite. |

## 4. Working in the page

| Key | Action |
|---|---|
| `E` | Error |
| `N` | No error |
| `→` / `←` | Next / previous claim |
| `U` | Next claim still to do |

Click a box to change a label at any time. The page does not move on by itself after a label, so you can add a note
first. **Labels and notes save as you go**: you can close the browser and come back, on the same computer and browser.

**Take a break every 10 claims.** The page reminds you. Tiredness shows up as disagreement between you and Mohamed.

**At the end of each sitting, press Download my labels.** It saves a file called `labels_hana_<date>.csv`, a backup
of everything so far. To carry on another computer, open your page there and use **Load labels…** with that file.

## 5. When you have finished

1. Check the counter says **50 / 50 labelled** (use **To do** to find any you missed).
2. Press **Download my labels**.
3. **Send the file to Parsa directly** (a private message or email). Do not post it in the group chat or the shared
   Drive folder, and do not show it to Mohamed or discuss any claim with them yet: the two sets must stay independent
   until both are in.

After both your files are in, Parsa merges them and you and Mohamed meet briefly to agree a label for each claim
where you differed, using the rules above. The other pair does the same for their half. You will not see which model wrote each claim or what the verifier said until the
results are computed.

## If something goes wrong

- **The red "not saving" box appears:** you are in a private window, or storage is off. Open the file in a normal
  window; if labels are already done, download them first.
- **"This file holds no labels for Hana":** you loaded someone else's file. Load your own.
- **"This file is from a different sample":** it is a file from an older version of the page. Ask Parsa.
- **Anything else:** message Parsa with the claim number (S0..) and what you saw.
