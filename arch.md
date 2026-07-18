# Final Architecture

# Nature-Inspired Augmentation Selection for Cyber Banking Data Awareness using Nature-Inspired Genetic Optimization and Low-Resource NLP

---

# Overview

The proposed framework is designed as an **adaptive augmentation selection system** rather than a conventional augmentation generation pipeline. Instead of blindly applying every augmentation technique to every complaint, the framework first analyzes the incoming cyber banking complaint, estimates its complexity, retrieves historical optimization knowledge, and intelligently selects the most appropriate augmentation strategy.

The objective is to maximize classification performance while minimizing unnecessary augmentation, semantic drift, and computational cost.

---

# Complete System Architecture

```text
                         Cyber Banking Complaint Dataset
                     (CFPB Complaint Subset - Runtime Stream)
                                      │
                                      ▼
                           Runtime Chunk Scheduler
                          (5–10 Complaints per Chunk)
                                      │
                                      ▼

=============================================================
                 PHASE 1 : COMPLAINT UNDERSTANDING
=============================================================

                      Text Cleaning & Preprocessing
                                   │
                                   ▼
                        Fraud Entity Extraction
                                   │
                                   ▼
                    Protected Entity Replacement
           [OTP] [CARD] [ACCOUNT] [UPI] [AMOUNT]
                                   │
                                   ▼
                     Fraud Graph Construction
                                   │
                                   ▼
                Fraud-Aware Complexity Index (FACI)

FACI Feature Vector

• Complexity
• Entity Density
• Fraud Density
• Risk Score
• Ambiguity
• Rare Word Density
• Semantic Entropy
• Recommended Budget

                                   │
                                   ▼

=============================================================
           PHASE 2 : KNOWLEDGE-DRIVEN POLICY RETRIEVAL
=============================================================

                        Policy Memory Database
                                   │
                                   ▼
             Retrieve Similar Historical Policies
                  (FACI Similarity Search)
                                   │
                                   ▼
               Initialize Elite Optimization Population

                                   │
                                   ▼

=============================================================
       PHASE 3 : NATURE-INSPIRED GENETIC OPTIMIZATION
=============================================================

                  Genetic Algorithm (GA)

             Global Exploration of Policy Space

Population Initialization
        │
Selection
        │
Crossover
        │
Mutation
        │
Elite Preservation
        │
Candidate Augmentation Policies

                                   │
                                   ▼

=============================================================
          PHASE 4 : GREY WOLF OPTIMIZATION (GWO)
=============================================================

                  Alpha
                  Beta
                  Delta
                  Omega

                        Local Policy Refinement

                                   │
                                   ▼

                    Optimal Augmentation Policy

                                   │
                                   ▼

=============================================================
            PHASE 5 : AUGMENTATION SELECTION
=============================================================

The optimizer DOES NOT generate data here.

It makes an intelligent decision:

──────────────────────────────
No Augmentation
──────────────────────────────
Back Translation
──────────────────────────────
BERT Contextual Augmentation
──────────────────────────────
Hybrid (BT + BERT)
──────────────────────────────

The optimizer also predicts

• Augmentation Budget

• Confidence Score

• Expected Utility

• Semantic Threshold

• Expected Computational Cost

                                   │
                                   ▼

                 Selected Augmentation Strategy

                                   │
                                   ▼

=============================================================
           PHASE 6 : AUGMENTATION GENERATION
=============================================================

IF Strategy == No Augmentation

        ↓

Skip Generation

ELSE

        ↓

Generate Selected Augmentation

• Back Translation

or

• BERT Contextual

or

• Hybrid BT + BERT

Generate only the required number of synthetic samples.

                                   │
                                   ▼

=============================================================
           PHASE 7 : SEMANTIC VALIDATION
=============================================================

Entity Preservation

↓

Semantic Similarity

↓

Duplicate Detection

↓

Redundancy Filter

↓

Label Consistency

↓

Accept / Reject Generated Samples

                                   │
                                   ▼

=============================================================
            PHASE 8 : INCREMENTAL LEARNING
=============================================================

Replay Buffer

↓

Incremental RoBERTa Training

↓

Evaluation

↓

Accuracy

Macro Precision

Macro Recall

Macro F1

Loss

                                   │
                                   ▼

=============================================================
      PHASE 9 : FEEDBACK & CONTINUAL OPTIMIZATION
=============================================================

Performance Analysis

↓

Utility Estimation

↓

Fitness Update

↓

Policy Memory Update

↓

GA Population Update

↓

GWO Wolf Update

↓

Next Runtime Chunk
```

---

# Detailed Workflow

## Phase 1 — Complaint Understanding

Every incoming complaint is first analyzed before any augmentation decision is made.

The preprocessing module performs:

- Text Cleaning
- Tokenization
- Banking Entity Detection
- Fraud Keyword Detection
- Protected Entity Masking

The complaint is transformed into a Fraud Graph describing relationships between fraud entities.

---

## Phase 2 — FACI Computation

The Fraud-Aware Complexity Index (FACI) estimates how difficult the complaint is to learn.

The FACI vector contains:

- Linguistic Complexity
- Entity Density
- Fraud Density
- Risk Score
- Semantic Entropy
- Rare Word Density
- Ambiguity
- Recommended Augmentation Budget

Rather than acting as a simple score, FACI becomes the primary feature representation guiding optimization.

---

## Phase 3 — Policy Memory

Before optimization begins, the framework searches the Policy Memory.

Instead of initializing optimization randomly, the system retrieves augmentation policies that previously performed well on complaints with similar FACI characteristics.

This enables continual optimization across runtime chunks.

---

## Phase 4 — Genetic Algorithm

The Genetic Algorithm performs global exploration.

Each chromosome represents a complete augmentation policy.

Example:

```text
Policy

{

Strategy,

Budget,

Back Translation Ratio,

BERT Ratio,

Semantic Threshold,

Entity Preservation Weight,

Confidence,

Expected Utility

}
```

The GA performs:

- Population Initialization
- Selection
- Crossover
- Mutation
- Elite Preservation

Its objective is to discover diverse candidate augmentation policies.

---

## Phase 5 — Grey Wolf Optimization

The elite chromosomes generated by the GA become the initial wolf population.

Grey Wolf Optimization performs local refinement.

The optimizer adjusts:

- Strategy Confidence
- Augmentation Budget
- Utility
- Semantic Threshold
- Cost

The Alpha Wolf represents the final augmentation policy.

---

# Augmentation Selection

This is the core contribution of the framework.

Instead of automatically generating augmentations, the optimizer first decides which augmentation strategy should be applied.

Possible decisions include:

- No Augmentation
- Back Translation
- BERT Contextual Augmentation
- Hybrid Back Translation + BERT

Example

```text
Complaint

↓

FACI

↓

Optimization

↓

Selected Strategy

↓

Generate Selected Augmentation
```

The framework therefore performs **selection before generation**.

---

# Augmentation Generation

Only the selected strategy is executed.

Examples:

Low Complexity

↓

No Augmentation

---

Medium Complexity

↓

BERT Contextual

---

High Complexity

↓

Back Translation

---

Very High Complexity

↓

Hybrid BT + BERT

---

This minimizes unnecessary synthetic data generation.

---

# Semantic Validation

Every generated sample passes through a validation stage.

Validation includes:

- Entity Preservation
- Label Consistency
- Semantic Similarity
- Duplicate Detection
- Redundancy Filtering

Only validated samples are accepted.

---

# Incremental Learning

Accepted samples are added to the replay buffer.

RoBERTa is incrementally updated using:

- Original Samples
- Accepted Augmented Samples

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- Macro F1
- Loss

---

# Feedback Loop

After evaluation, the framework updates:

- Policy Memory
- Genetic Population
- Grey Wolf Positions
- Historical Utility
- Optimization Statistics

The optimizer therefore improves continuously over runtime chunks.

---

# Research Contribution

Unlike conventional NLP augmentation methods that indiscriminately apply augmentation to every training sample, the proposed framework introduces an intelligent **Augmentation Selection Engine**.

The framework:

- Understands the complaint.
- Estimates fraud complexity using FACI.
- Retrieves historical optimization knowledge.
- Uses a hybrid Genetic Algorithm and Grey Wolf Optimization framework to search and refine augmentation policies.
- Selects the most appropriate augmentation strategy.
- Generates only the required synthetic samples.
- Validates semantic correctness.
- Learns incrementally through continual feedback.

This transforms data augmentation from a fixed preprocessing step into an adaptive, explainable, and optimization-driven decision-making process for low-resource cyber banking NLP.



# data sample:
$ head -n 20 data/complaints.json
[
{"date_received":"2020-07-06","product":"Credit reporting, credit repair services, or other personal consumer reports","sub_product":"Credit reporting","issue":"Incorrect information on your report","sub_issue":"Information belongs to someone else","complaint_what_happened":"","company_public_response":"Company has responded to the consumer and the CFPB and chooses not to provide a public response","company":"Experian Information Solutions Inc.","state":"FL","zip_code":"346XX","tags":"","consumer_consent_provided":"Other","submitted_via":"Web","date_sent_to_company":"2020-07-06","company_response":"Closed with explanation","timely":"Yes","consumer_disputed":"N/A","complaint_id":"3730948"},
{"date_received":"2019-12-26","product":"Credit card or prepaid card","sub_product":"General-purpose credit card or charge card","issue":"Advertising and marketing, including promotional offers","sub_issue":"Confusing or misleading advertising about the credit card","complaint_what_happened":"","company_public_response":"","company":"CAPITAL ONE FINANCIAL CORPORATION","state":"CA","zip_code":"94025","tags":"","consumer_consent_provided":"Consent not provided","submitted_via":"Web","date_sent_to_company":"2019-12-26","company_response":"Closed with explanation","timely":"Yes","consumer_disputed":"N/A","complaint_id":"3477549"},
{"date_received":"2020-05-08","product":"Credit reporting, credit repair services, or other personal consumer reports","sub_product":"Credit reporting","issue":"Incorrect information on your report","sub_issue":"Information belongs to someone else","complaint_what_happened":"These are not my accounts.","company_public_response":"Company has responded to the consumer and the CFPB and chooses not to provide a public response","company":"Experian Information Solutions Inc.","state":"NV","zip_code":"89030","tags":"","consumer_consent_provided":"Consent provided","submitted_via":"Web","date_sent_to_company":"2020-05-08","company_response":"Closed with explanation","timely":"Yes","consumer_disputed":"N/A","complaint_id":"3642453"},
{"date_received":"2024-01-05","product":"Credit reporting or other personal consumer reports","sub_product":"Credit reporting","issue":"Incorrect information on your report","sub_issue":"Information belongs to someone else","complaint_what_happened":"Kindly address this issue on my credit report. I assert that this account is not mine and believe it to be fraudulent. I urge you to correct this mistake and have provided supporting documents for verification.","company_public_response":"Company has responded to the consumer and the CFPB and chooses not to provide a public response","company":"Experian Information Solutions Inc.","state":"IL","zip_code":"60502","tags":"","consumer_consent_provided":"Consent provided","submitted_via":"Web","date_sent_to_company":"2024-01-05","company_response":"Closed with non-monetary relief","timely":"Yes","consumer_disputed":"N/A","complaint_id":"8113747"},
{"date_received":"2024-01-21","product":"Credit reporting or other personal consumer reports","sub_product":"Credit reporting","issue":"Improper use of your report","sub_issue":"Credit inquiries on your report that you don't recognize","complaint_what_happened":"","company_public_response":"Company has responded to the consumer and the CFPB and chooses not to provide a public response","company":"Experian Information Solutions Inc.","state":"NC","zip_code":"27401","tags":"Servicemember","consumer_consent_provided":"Consent not provided","submitted_via":"Web","date_sent_to_company":"2024-01-21","company_response":"Closed with explanation","timely":"Yes","consumer_disputed":"N/A","complaint_id":"8191825"},        
{"date_received":"2020-03-19","product":"Credit reporting, credit repair services, or other personal consumer reports","sub_product":"Credit reporting","issue":"Incorrect information on your report","sub_issue":"Information belongs to someone else","complaint_what_happened":"I wrote three requests, the unverified accounts listed below still remain on my credit report in violation of Federal Law. Equifax is under FCRA law to obtain the of the original creditors documentation on file to verify that this information is mine and is correct. I have already filed a FTC Report and Police Report. Who verified these accounts? You have NOT provided me a copy of ANY original documentation ( a consumer contract with my signature on it ) as required under Section 609 ( a ) ( 1 ) ( A ) & Section 611 ( a ) ( 1 ) ( A ). Furthermore you have failed to provide the method of verification as required under Section 611 ( a ) ( 7 ). Please be advised that under Section 611 ( 5 ) ( A ) of the FCRA you are required to promptly DELETE all information which can not be verified. \nThe law is very clear as to the Civil liability and the remedy available to me ( Section 616 & 617 ) if you fail to comply with Federal Law. I am a litigious consumer and fully intend on pursuing litigation in this matter to enforce my rights under the FCRA. Please remove the following accounts : XXXX XXXX XXXX XXXX XXXX, XXXX XXXX XXXX   XXXX","company_public_response":"","company":"EQUIFAX, INC.","state":"NC","zip_code":"28562","tags":"","consumer_consent_provided":"Consent provided","submitted_via":"Web","date_sent_to_company":"2020-03-19","company_response":"Closed with explanation","timely":"Yes","consumer_disputed":"N/A","complaint_id":"3573294"},
{"date_received":"2019-10-22","product":"Credit reporting, credit repair services, or other personal consumer reports","sub_product":"Credit reporting","issue":"Incorrect information on your report","sub_issue":"Old information reappears or never goes away","complaint_what_happened":"XXXX XXXX has a old account settled in XXXX that keeps reappearing on my experian credit report and it now states its a debt from XXXX settled and XXXX balance. Its a total lie and incorrect. It was from XXXX settled way back then. It needs to be removed. Please make them remove it. Its effecting my score.","company_public_response":"Company has responded to the consumer and the CFPB and chooses not to provide a public response","company":"Experian Information Solutions Inc.","state":"HI","zip_code":"967XX","tags":"Servicemember","consumer_consent_provided":"Consent provided","submitted_via":"Web","date_sent_to_company":"2019-10-22","company_response":"Closed with non-monetary relief","timely":"Yes","consumer_disputed":"N/A","complaint_id":"3414709"},
{"date_received":"2020-03-29","product":"Student loan","sub_product":"Federal student loan servicing","issue":"Dealing with your lender or servicer","sub_issue":"Keep getting calls about your loan","complaint_what_happened":"They call at all hours and on the weekends using various numbers.","company_public_response":"","company":"Navient Solutions, LLC.","state":"CA","zip_code":"92028","tags":"Servicemember","consumer_consent_provided":"Consent provided","submitted_via":"Web","date_sent_to_company":"2020-03-29","company_response":"Closed with explanation","timely":"Yes","consumer_disputed":"N/A","complaint_id":"3584679"},
{"date_received":"2024-01-23","product":"Credit reporting or other personal consumer reports","sub_product":"Credit reporting","issue":"Problem with a company's investigation into an existing problem","sub_issue":"Their investigation did not fix an error on your report","complaint_what_happened":"","company_public_response":"","company":"EQUIFAX, INC.","state":"IN","zip_code":"46222","tags":"","consumer_consent_provided":"Consent not provided","submitted_via":"Web","date_sent_to_company":"2024-01-23","company_response":"Closed with explanation","timely":"Yes","consumer_disputed":"N/A","complaint_id":"8207223"},
{"date_received":"2019-12-20","product":"Checking or savings account","sub_product":"Other banking product or service","issue":"Managing an account","sub_issue":"Funds not handled or disbursed as instructed","complaint_what_happened":"","company_public_response":"Company has responded to the consumer and the CFPB and chooses not to provide a public response","company":"WELLS FARGO & COMPANY","state":"FL","zip_code":"33064","tags":"","consumer_consent_provided":"N/A","submitted_via":"Referral","date_sent_to_company":"2019-12-23","company_response":"Closed with explanation","timely":"Yes","consumer_disputed":"N/A","complaint_id":"3475858"},
{"date_received":"2020-03-17","product":"Credit reporting, credit repair services, or other personal consumer reports","sub_product":"Credit reporting","issue":"Incorrect information on your report","sub_issue":"Information belongs to someone else","complaint_what_happened":"I was recently going to check out a new car at XXXX. After carefully consideration I decided to apply for a loan for this car. Moments later I was decline for a lower interest rate and approved for an extremely high interest rate. Which promoted me to check my credit report. After doing so I notice a couple of items on my report that are not mine. These items need to be deleted from my credit report. \n\nXXXX XXXX XXXX {$140.00} XXXX  XXXX {$450.00} XXXX XXXX XXXX {$7800.00} XXXX   XXXX  XXXX {$370.00} XXXX XXXX {$10000.00}","company_public_response":"Company has responded to the consumer and the CFPB and chooses not to provide a public response","company":"TRANSUNION INTERMEDIATE HOLDINGS, INC.","state":"AR","zip_code":"72211","tags":"","consumer_consent_provided":"Consent provided","submitted_via":"Web","date_sent_to_company":"2020-03-17","company_response":"Closed with explanation","timely":"Yes","consumer_disputed":"N/A","complaint_id":"3569824"},
{"date_received":"2024-01-29","product":"Credit reporting or other personal consumer reports","sub_product":"Credit reporting","issue":"Incorrect information on your report","sub_issue":"Information belongs to someone else","complaint_what_happened":"","company_public_response":"Company has responded to the consumer and the CFPB and chooses not to provide a public response","company":"TRANSUNION INTERMEDIATE HOLDINGS, INC.","state":"IL","zip_code":"60411","tags":"","consumer_consent_provided":"Other","submitted_via":"Web","date_sent_to_company":"2024-01-29","company_response":"Closed with non-monetary relief","timely":"Yes","consumer_disputed":"N/A","complaint_id":"8242689"},
{"date_received":"2020-05-18","product":"Credit reporting, credit repair services, or other personal consumer reports","sub_product":"Credit reporting","issue":"Incorrect information on your report","sub_issue":"Information belongs to someone else","complaint_what_happened":"","company_public_response":"Company has responded to the consumer and the CFPB and chooses not to provide a public response","company":"Experian Information Solutions Inc.","state":"IN","zip_code":"46383","tags":"","consumer_consent_provided":"N/A","submitted_via":"Phone","date_sent_to_company":"2020-05-18","company_response":"Closed with explanation","timely":"Yes","consumer_disputed":"N/A","complaint_id":"3656557"},
{"date_received":"2019-09-23","product":"Credit reporting, credit repair services, or other personal consumer reports","sub_product":"Credit reporting","issue":"Incorrect information on your report","sub_issue":"Information belongs to someone else","complaint_what_happened":"","company_public_response":"Company has responded to the consumer and the CFPB and chooses not to provide a public response","company":"TRANSUNION INTERMEDIATE HOLDINGS, INC.","state":"NY","zip_code":"11239","tags":"","consumer_consent_provided":"Consent not provided","submitted_via":"Web","date_sent_to_company":"2019-09-23","company_response":"Closed with explanation","timely":"Yes","consumer_disputed":"N/A","complaint_id":"3382961"},     
{"date_received":"2014-05-22","product":"Bank account or service","sub_product":"Checking account","issue":"Account opening, closing, or management","sub_issue":"","complaint_what_happened":"","company_public_response":"","company":"WELLS FARGO & COMPANY","state":"GA","zip_code":"30083","tags":"","consumer_consent_provided":"N/A","submitted_via":"Postal mail","date_sent_to_company":"2014-05-27","company_response":"Closed with explanation","timely":"Yes","consumer_disputed":"No","complaint_id":"864381"},       
{"date_received":"2019-11-18","product":"Credit card or prepaid card","sub_product":"General-purpose credit card or charge card","issue":"Problem with a purchase shown on your statement","sub_issue":"Credit card company isn't resolving a dispute about a purchase on your statement","complaint_what_happened":"XXXX claimed they delivered a package to my address, which I never received Listed below are the delivery facts from the USPS regarding shipments XXXX & XXXX, with the supporting documentation enclosed. \n\nI. There was no signature II. The packages were delivered to zip code XXXX. \na. There are XXXX souls living in XXXX dwelling units in ZIP code XXXX. \n\nXXXX provided no information that the package was delivered to my address, let alone signed by anyone. The only information provided shows that the packages were left at or near the mailbox in one of XXXX dwelling units. At or near mailbox does not indicate that the package was even placed inside of a secured area, let alone a secured area in my building. \n\nHow am I being held responsible for a package for which there is no proof that I received? XXXX for a small fee, {$.00}, could have requested delivery confirmation from the USPS. This would have ensured that an authorized agent signed for the package. Alternately, XXXX could have ensured the packages for approximately {$4.00} each. \n\nThere is no evidence to support that I received the packages.","company_public_response":"","company":"DISCOVER BANK","state":"MA","zip_code":"021XX","tags":"","consumer_consent_provided":"Consent provided","submitted_via":"Web","date_sent_to_company":"2019-11-18","company_response":"Closed with explanation","timely":"Yes","consumer_disputed":"N/A","complaint_id":"3442136"},    
{"date_received":"2025-04-27","product":"Credit reporting or other personal consumer reports","sub_product":"Credit reporting","issue":"Incorrect information on your report","sub_issue":"Information belongs to someone else","complaint_what_happened":"","company_public_response":"Company has responded to the consumer and the CFPB and chooses not to provide a public response","company":"Experian Information Solutions Inc.","state":"CT","zip_code":"06517","tags":"","consumer_consent_provided":"Consent not provided","submitted_via":"Web","date_sent_to_company":"2025-04-27","company_response":"Closed with explanation","timely":"Yes","consumer_disputed":"N/A","complaint_id":"13203310"},
{"date_received":"2020-06-05","product":"Checking or savings account","sub_product":"Checking account","issue":"Managing an account","sub_issue":"Problem using a debit or ATM card","complaint_what_happened":"","company_public_response":"Company has responded to the consumer and the CFPB and chooses not to provide a public response","company":"CITIBANK, N.A.","state":"NY","zip_code":"10466","tags":"","consumer_consent_provided":"Consent not provided","submitted_via":"Web","date_sent_to_company":"2020-06-05","company_response":"Closed with explanation","timely":"Yes","consumer_disputed":"N/A","complaint_id":"3684669"},
{"date_received":"2019-03-17","product":"Credit reporting, credit repair services, or other personal consumer reports","sub_product":"Credit reporting","issue":"Incorrect information on your report","sub_issue":"Account status incorrect","complaint_what_happened":"","company_public_response":"Company has responded to the consumer and the CFPB and chooses not to provide a public response","company":"Experian Information Solutions Inc.","state":"CA","zip_code":"91723","tags":"","consumer_consent_provided":"Consent not provided","submitted_via":"Web","date_sent_to_company":"2019-03-17","company_response":"Closed with non-monetary relief","timely":"Yes","consumer_disputed":"N/A","complaint_id":"3182426"},
]