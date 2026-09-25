# ECOM6004 Assessment 1 — Semester 2, 2026

This folder contains the assessment brief, datasets, data dictionaries, rubric
and Gen-AI disclosure form.

## Submit

- One report knitted or compiled to PDF or HTML. The report must contain only
  the reported results, their interpretation and discussion, and the reference
  list. The expected PDF length is 15--20 pages, including the reference list;
  a complete report may be shorter but must not exceed 20 pages. An HTML report
  should provide the same substantive content at a similarly concise length.
- The complete R Markdown, Quarto, notebook or other source file, submitted
  separately from the report.
- Any supporting R or Python code, submitted separately from the report. Do not
  include code or a code appendix in the report.
- The completed three-page `ECOM6004 Assessment 1 - Student Gen-AI Disclosure Form - Student_Name.docx`, renamed with your student ID and submitted separately as instructed in the brief. This form is the complete disclosure.
- GenAI use is limited to the support activities permitted in the assessment
  brief. Every use must be disclosed, and students must conduct the analysis
  and write the report themselves.
- Your submitted analysis, code, interpretations and writing must be completed
  individually and must be your own.

## Choose one dataset

- Bank: `bank_assessment_sem2_2026.csv` and `bank_dictionary.txt`.
- Telco: `telco_assessment_sem2_2026.csv` and `telco_dictionary.txt`.

## Analysis sequence

- After the dictionary-specified data-quality checks, type conversions, coding
  and transformations, create one reproducible 80% training and 20% test split
  stratified by the binary outcome. Use your full numeric student ID as the
  random seed and report the resulting row counts and event rates.
- Question 2 fits, compares and selects the binary model using your training
  sample only. Keep its selected specification and estimated training
  coefficients unchanged.
- Question 3 applies that unchanged binary model once to your held-out test
  sample, then completes the binary evaluation and client recommendation. Do
  not revise Question 2 after seeing test outcomes.
- Question 4 fits and interprets ordered logit and ordered probit using the full
  dataset after the required preparation.
- Question 5 completes the reproducibility record, decision log and separate Gen-AI disclosure requirements at the end of the assessment.


See `SOFTWARE_SETUP.md` for the minimal package requirements.
