from Process_Input import conclusion
import os
import re
import json
import pandas as pd
from tqdm import tqdm

def extract_fields(text):
    """
    Extracts fields from the given text.
    Assumes the text has the format:
    'Yes. <Conference>. Reasoning: <Rationale>'
    """
    publishable = 1 if text.startswith("Yes") else 0  # Determine publishable
    conference_match = re.search(r"Yes\. (.+?)\. Reasoning:", text)
    conference = conference_match.group(1).strip() if conference_match else ""
    rationale_match = re.search(r"(Reason(?:ing)?:|Yes\. .*?\.)\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    rationale = rationale_match.group(2).strip() if rationale_match else "No rationale provided"
    
    return publishable, conference, rationale

def append_to_dataframe(output_text, df, paper_id):
    """
    Appends extracted fields to the given Pandas DataFrame.
    """
    # Extract fields from the given text
    publishable, conference, rationale = extract_fields(output_text)
    if 'Reasoning:' in rationale:
        rationale = rationale.replace('Reasoning:', '')
    rationale = rationale.strip()
    # Create a new row as a dictionary
    new_row = {
        "Paper ID": paper_id,
        "Publishable": publishable,
        "Conference": conference,
        "Rationale": rationale
    }

    with open('output.txt', 'a') as f:
        f.write(json.dumps(new_row))
        f.write('\n')
    # Append the new row using loc (modifies DataFrame in place)
    df.loc[len(df)] = [paper_id, publishable, conference, rationale]
    return

files = os.listdir('Papers')
total_files = len(files)
output_file = pd.DataFrame(columns = ['Paper ID', 'Publishable', 'Conference', 'Rationale'])

with tqdm(files, desc="Processing Papers", unit="file", total=total_files, 
              bar_format="{l_bar}{bar}| {n}/{total} files") as pbar:
        for file in pbar:
            doc_path = os.path.join('Papers', file)
            output = conclusion(doc_path)
            append_to_dataframe(output, output_file, file.replace('.pdf', ''))

output_file.to_csv('results.csv', index = False)
