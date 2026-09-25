ECOM6004 ASSESSMENT 1 DATA DICTIONARY - BANK MARKETING

Assessment file: bank_assessment_sem2_2026.csv
Dimensions: 45,211 rows and 20 columns

1.⁠ ⁠Relevant information

Each row records a bank client and the current telephone-marketing campaign.
The binary outcome y indicates whether the client subscribed to a term deposit.

2.⁠ ⁠Source variables

Client characteristics

  age        Age in years (numeric).
  job        Job category: admin., unknown, unemployed, management,
             housemaid, entrepreneur, student, blue-collar, self-employed,
             retired, technician or services.
  marital    Marital status: married, divorced or single. Divorced includes
             clients recorded as divorced or widowed.
  education  Education category: unknown, secondary, primary or tertiary.
  default    Credit-default indicator: yes or no.
  balance    Average yearly balance in euros (numeric).
  housing    Housing-loan indicator: yes or no.
  loan       Personal-loan indicator: yes or no.

Current campaign contact

  contact    Contact method: unknown, telephone or cellular.
  day        Day of the month of the most recent contact (numeric).
  month      Month of the most recent contact: jan, feb, mar, apr, may, jun,
             jul, aug, sep, oct, nov or dec.
  duration   Duration of the most recent contact in seconds (numeric). This is
             observed only after the call has occurred.
  campaign   Number of contacts made during the current campaign, including
             the most recent contact (numeric).

Previous campaign history

  pdays      Days since the client was last contacted in a previous campaign
             (numeric). The value -1 is a status code meaning that the client
             was not previously contacted; it is not minus one elapsed day.
  previous   Number of contacts made before the current campaign (numeric).
  poutcome   Previous campaign outcome: unknown, other, failure or success.

Outcome

  y          Term-deposit subscription: yes or no.

The source variables contain no missing values.

3.⁠ ⁠Assessment fields

  .row_id                 Unique row identifier.
  response_level          Ordered response:
                            0 = no subscription;
                            1 = subscription after two or more contacts in the
                                current campaign;
                            2 = subscription on the first contact.
                          Thus y = "yes" if and only if response_level > 0.
                          The ordered categories are formed from y and campaign.
  response_level_label    Readable label for response_level.

4.⁠ ⁠Required preparation for the common specification

  target       1 when y = "yes" and 0 otherwise.
  age10        (age - 40) / 10. One unit represents ten years relative to
               age 40.
  balance1000  balance / 1000. One unit represents EUR 1,000.
  prior_status Construct from pdays and poutcome as follows:
                 "not previously contacted" when pdays = -1;
                 "previous success" when pdays != -1 and
                   poutcome = "success";
                 "previous failure" when pdays != -1 and
                   poutcome = "failure";
                 "previous other/unknown" otherwise.

Reference categories

  housing      no
  loan         no
  contact      telephone
  prior_status not previously contacted

5.⁠ ⁠Citation

Moro, S., Rita, P., and Cortez, P. (2014). Bank Marketing [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5K306