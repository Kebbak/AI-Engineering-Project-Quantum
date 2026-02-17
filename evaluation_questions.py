# Evaluation questions for RAG LLM system
# Each question relates to a company policy in the corpus directory
# You can use this file to run automated or manual evaluation

questions = [
    # PTO Policy
    "How many PTO days can employees carry over to the next year?",
    "What is the monthly PTO accrual rate?",
    "Who should employees contact with PTO questions?",
    "How far in advance should PTO be requested?",

    # Security Policy
    "What should employees do if they suspect a security incident?",
    "Are employees required to complete security training?",
    "What are the password requirements for company accounts?",
    "Who should be contacted for security policy questions?",

    # Expense Policy
    "What types of expenses are eligible for reimbursement?",
    "What is the deadline for submitting expense reports?",
    "Is manager approval required for reimbursements?",
    "Who should employees contact for expense policy questions?",

    # Remote Work Policy
    "How many days per week can employees work remotely?",
    "What are the expectations for remote workers regarding meetings?",
    "What security measures must remote employees follow?",
    "Who is eligible for remote work?",

    # Holiday Policy
    "Which holidays are observed by the company?",
    "What happens if a holiday falls on a weekend?",
    "How many floating holidays can employees select per year?",
    "Who should be contacted for holiday policy questions?",

    # IT Usage Policy
    "What should employees do if a device is lost or stolen?",
    "Are employees allowed to install unauthorized software?",
    "What are the rules for password management?",
    "Who should employees contact for IT policy questions?",

    # Code of Conduct
    "What should employees do if they witness harassment?",
    "What are the expectations for workplace behavior?",
    "Who should be contacted for code of conduct questions?",

    # Travel Policy
    "How should employees book business travel?",
    "What is the daily meal allowance policy for travel?",
    "Who should employees contact for travel policy questions?"
]

# refuse to answer questions that are not in the list above such as
# "What is the company's policy on cryptocurrency trading for employees?"
# "What is the company's policy on social media use for employees?"
