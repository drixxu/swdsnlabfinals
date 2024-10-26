import inquirer
import numpy as np 

# Input the the details
def get_user_input():
    questions = [
        inquirer.Text("YEAR", message="Year"),
        inquirer.Text("MONTH", message="Month (1-12)"),
        inquirer.Text("ITEM CODE", message="Item Code"),
        inquirer.Text("RETAIL TRANSFERS", message="Retail Transfers (can be decimal)"),
        inquirer.Text("WAREHOUSE SALES", message="Warehouse Sales (can be decimal)")
    ]

    # Prompt 
    answers = inquirer.prompt(questions)

    # Convert the answers
    new_data = np.array(
        [[
            int(answers['YEAR']),
            int(answers['MONTH']),
            int(answers['ITEM CODE']),
            float(answers['RETAIL TRANSFERS']),
            float(answers['WAREHOUSE SALES'])
        ]]
    )

    return new_data
