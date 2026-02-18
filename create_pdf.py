# This script creates a PDF document containing the workplace safety policy.
from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Workplace Safety Policy", ln=True, align="C")
pdf.ln(10)
pdf.multi_cell(0, 10, txt="Overview:\nThe company is committed to providing a safe and healthy workplace for all employees.\n\nGeneral Guidelines:\n- Follow all posted safety instructions.\n- Report hazards or unsafe conditions to your supervisor immediately.\n- Participate in required safety training.\n\nEmergency Procedures:\n- Know the location of emergency exits and equipment.\n- In case of fire, evacuate the building and assemble at the designated area.\n\nReferences:\nContact the Safety Officer for questions.")
pdf.output("corpus/workplace_safety_policy.pdf")
print("PDF created: corpus/workplace_safety_policy.pdf")
