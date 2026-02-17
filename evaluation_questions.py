# Evaluation questions for RAG LLM system
# Each question relates to a company policy in the corpus directory
# You can use this file to run automated or manual evaluation

questions = [
#PTO Policy

"What is the monthly PTO accrual rate for employees?",
"For what purposes can PTO be used?",
"How far in advance must PTO requests be submitted?",
"Is manager approval required for PTO requests?",
"How many unused PTO days can be carried over to the next year?",

# Security Policy
"What are the requirements for passwords under the security policy?",
"Is sharing login credentials allowed?",
"How should sensitive data be protected?",
"What should employees do if they encounter a security incident?",
"Is annual security training required for employees?",

# Expense Policy
"What types of expenses are eligible for reimbursement?",
"Is manager approval required for all reimbursements?",
"How soon must expense reports be submitted after the expense?",
"What documentation must be included with expense reports?",

# Remote Work Policy
"How many days per week can employees work remotely?",
"Is manager approval required for remote work?",
"Who is eligible for remote work?",
"What security measures must remote employees follow?",

# Holiday Policy
"Which holidays are observed as paid holidays by the company?",
"How many floating holidays can employees select per year?",

]

# refuse to answer questions that are not in the list above such as
# "What is the company's policy on cryptocurrency trading for employees?"
# "What is the company's policy on social media use for employees?"
