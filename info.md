# What is This Project?

## Simple Explanation

This project is a **decision-making tool** for a bank. It helps the bank answer one important question:

**"Which customers should we contact about term deposits?"**

A term deposit is like a savings account where you put money away for a fixed time and earn interest. The bank wants to know which customers are most likely to say "yes" when contacted, so they don't waste time calling people who aren't interested.

---

## The Problem

Imagine you work at a bank's marketing department. You have a list of thousands of customers. You want to call them to offer a term deposit, but:

- Calling everyone is too expensive and time-consuming
- Some customers will definitely say "yes"
- Some customers will definitely say "no"
- Many customers are somewhere in between

**The challenge**: How do you figure out who to call first, and who to skip?

---

## The Solution

This project uses **customer data** to make smart predictions about who is likely to subscribe to a term deposit. It looks at information like:

- **Age**: How old is the customer?
- **Balance**: How much money do they have in their account?
- **Housing loan**: Do they have a home loan?
- **Personal loan**: Do they have other loans?
- **Contact method**: Should we call their phone or send a text?
- **Past history**: Have we contacted them before? Did it work?

Using this information, the project builds **mathematical models** that predict the likelihood of each customer saying "yes" to a term deposit offer.

---

## How It Works (Step by Step)

### Step 1: Getting the Data

The project starts with a file containing information about thousands of bank customers. This is like a big spreadsheet with details about each person.

### Step 2: Cleaning and Preparing the Data

Before using the data, the project checks it for problems:
- Are there any missing pieces of information?
- Are there any duplicate entries?
- Are the numbers in the right format?

It also transforms the data to make it easier to work with:
- Converts ages into a standardized format
- Converts account balances into thousands (so $5,000 becomes 5)
- Creates categories for past contact history

### Step 3: Splitting the Data

The project divides the customer list into two groups:

- **Training group (80%)**: Used to teach the computer patterns
- **Test group (20%)**: Used to check if the computer learned correctly

This is like studying for a test - you practice on some questions (training), then take a final exam on different questions (testing) to see if you really learned.

### Step 4: Building Prediction Models

The project creates three different mathematical models to predict customer behavior:

1. **Linear Probability Model**: A simple approach that calculates the direct probability
2. **Logit Model**: A more sophisticated approach using logarithms
3. **Probit Model**: Another sophisticated approach using probability curves

All three models look at the same customer information but use slightly different mathematical methods to make predictions.

### Step 5: Testing the Models

After building the models, the project tests them using the test group (the 20% of customers the models haven't seen before). It checks:

- **Accuracy**: How often did the model correctly predict who would say "yes"?
- **Brier Score**: A measure of how reliable the predictions are (lower is better)
- **Sensitivity**: How good is the model at finding customers who will say "yes"?
- **Specificity**: How good is the model at identifying customers who will say "no"?

### Step 6: Making a Decision

Based on the test results, the project creates a **decision log** that answers:

- Which model performed best?
- Should the bank proceed with using this model to guide contact decisions?
- What is the reasoning behind this recommendation?

The decision is cautious - if the model isn't accurate enough (less than 60% balanced accuracy), the project recommends using additional human judgment rather than relying solely on the model.

### Step 7: Understanding Customer Efficiency

The project also looks at a more detailed question: **"How efficient is each customer?"**

Instead of just "yes" or "no", it categorizes customers into three levels:

- **Level 0**: Customer won't subscribe at all
- **Level 1**: Customer will subscribe, but only after being contacted multiple times
- **Level 2**: Customer will subscribe on the very first contact (most efficient!)

This helps the bank prioritize customers who are "easy wins" - those who will say yes quickly without requiring many follow-up calls.

### Step 8: Creating a Record

Finally, the project creates a **reproducibility record**. This is like a recipe card that documents:

- Which student ID was used (to ensure results can be reproduced)
- Which mathematical formulas were used
- How the data was split
- What settings were used

This ensures that if someone else runs the same analysis with the same student ID, they'll get exactly the same results.

---

## What You Get as Output

When you run this project, it produces several pieces of information:

### 1. Data Audit Summary
A comprehensive report showing the quality of the data:
- How many customers are in the database
- Whether there are any missing or duplicate entries
- The distribution of customers who said "yes" vs "no"
- The age and balance ranges
- Validation checks for all numeric and categorical variables
- Consistency checks between related variables

### 2. Model Comparison Table
A table showing the results from all three models (LPM, Logit, Probit):
- Which factors are most important in predicting customer behavior
- Whether each factor has a positive or negative effect
- Whether the results are statistically significant (not just random chance)

### 3. Test Performance Metrics
Numbers showing how well the chosen model performs:
- Brier score (reliability of predictions)
- Balanced accuracy (overall correctness)
- Sensitivity (ability to find "yes" customers)
- Specificity (ability to find "no" customers)

### 4. Decision Log
A clear recommendation:
- Whether to proceed with using the model
- The reasoning behind the decision
- Key performance metrics that support the decision

### 5. Ordered Model Results
Information about customer efficiency:
- Which factors make customers more likely to subscribe quickly
- Comparison between different mathematical approaches
- Diagnostics showing whether the models worked correctly

### 6. Reproducibility Record
A complete record of how the analysis was done, ensuring it can be exactly reproduced later.

---

## Why This Matters

### For the Bank

- **Cost Savings**: By focusing on customers likely to say "yes", the bank saves money on marketing
- **Better Customer Experience**: Customers who aren't interested won't be bothered with unwanted calls
- **Data-Driven Decisions**: Instead of guessing, the bank uses actual data to guide decisions
- **Efficiency**: Understanding which customers convert quickly helps prioritize outreach

### For the Analysis

- **Rigorous Testing**: By using a separate test group, we ensure the models actually work, not just memorize the training data
- **Multiple Approaches**: Using three different models helps find the best approach
- **Transparency**: Every step is documented and reproducible
- **Caution**: The decision log ensures models are only used when they're accurate enough

---

## Who Is This For?

This project is designed for:

### Students and Researchers
- Learning how to build prediction models
- Understanding how to test and validate models
- Practicing data analysis techniques
- Completing academic assessments (ECOM6004 at Curtin University)

### Business Analysts
- Understanding how data can guide marketing decisions
- Learning about customer segmentation
- Seeing how to measure model performance
- Making data-driven recommendations

### Anyone Interested in Data Science
- Seeing a real-world example of predictive modeling
- Understanding the complete workflow from data to decision
- Learning about different types of statistical models
- Seeing how to ensure results are reproducible

---

## Key Concepts Explained Simply

### What is a "Model"?

Think of a model like a recipe. Just as a recipe tells you how ingredients combine to make a cake, a mathematical model tells you how customer characteristics combine to predict behavior.

### What is "Training"?

Training is like teaching a child to recognize animals. You show them pictures of cats and dogs, tell them which is which, and they learn the patterns. Similarly, we show the model customer data and tell it who said "yes" or "no", and it learns the patterns.

### What is "Testing"?

Testing is like giving the child a quiz with new pictures they haven't seen before. This checks if they really learned to recognize animals, or just memorized the practice pictures. Similarly, we test the model on customers it hasn't seen to check if it really learned to predict behavior.

### What is "Statistical Significance"?

This is a way of asking: "Is this result real, or just luck?" For example, if you flip a coin 10 times and get 7 heads, that might be luck. But if you flip it 1,000 times and get 700 heads, that's statistically significant - the coin is probably weighted. The project checks if the patterns it finds are real or just random chance.

### What is "Reproducibility"?

Reproducibility means that if someone else follows the same steps with the same data, they'll get the exact same results. This is crucial in science and research - it proves that the results are real and not just a fluke.

---

## The Big Picture

This project is about **turning data into decisions**. It takes raw information about bank customers, processes it carefully, builds mathematical models to understand patterns, tests those models rigorously, and provides clear recommendations for action.

The goal isn't just to predict who will say "yes" - it's to give the bank a reliable, tested tool they can use to make better decisions about who to contact, saving money and improving customer satisfaction.

---

## What Makes This Work?

### 1. Good Data
The project starts with clean, comprehensive data about real bank customers. Without good data, no model can work well.

### 2. Proper Preparation
The data is carefully cleaned and transformed before use. This is like washing and chopping vegetables before cooking - essential for a good result.

### 3. Rigorous Testing
The models are tested on data they haven't seen before. This ensures they actually learned patterns, not just memorized specific examples.

### 4. Multiple Approaches
Using three different models helps find the best approach and provides confidence in the results.

### 5. Clear Decision-Making
The project doesn't just give predictions - it provides a clear recommendation with reasoning.

### 6. Reproducibility
Every step is documented so the analysis can be exactly reproduced, which is essential for credibility.

---

## Summary

**In one sentence**: This project helps a bank use customer data to predict which customers are likely to subscribe to term deposits, so the bank can make smarter, more efficient contact decisions.

**In one paragraph**: This project takes customer information from a bank, uses mathematical models to predict who will say "yes" to term deposit offers, tests those models to ensure they work correctly, and provides clear recommendations about which customers to contact. It also analyzes how efficient each customer is (whether they need many contacts or just one), helping the bank prioritize the most promising leads.

**The key insight**: By using data and mathematical models, the bank can move from guessing which customers to contact to making informed, data-driven decisions that save money and improve results.

---

**End of Overview**
