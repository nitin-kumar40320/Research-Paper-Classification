# Research Paper Classifier
This project is meant to classify the research papers as **Publishable** or **Non-Publishable**. If the paper is publishable, it further suggests the conferences in which the paper may be submitted, with reasoning for the decision. The project involves the integration of LLM API calling and RAG.

# Dataset Information
The dataset employed was retrieved from Khadagpur Data Science Hackathon (KDSH), 2025 organized by [Khadagpur Data Analytics Group](https://www.kdagiitkgp.com/).
It consisted for 150 papers, of which 15 were labelled. These papers were to be classified as publishable or non-publishable.
If the paper is publishable, then its publishing conference among **CVPR**, **EMNLP**, **KDD**, **NeurIPS** and **TMLR** was given.

# Operation Instruction
- Keep the 'References' folder, 'Papers' folder, 'reference_data.csv' file and the Python code files within the same directory or you might need to change the directory path to the folders in the code files is they are kept elsewhere.
- Replace 'API_KEY_HERE' with a Gemini API key on line 61 in '[Process_Input.py](https://github.com/Devansh-arora02/Research_Paper_Classifier/blob/main/Process_Input.py)'.
- Replace 'API_KEY_HERE' with a HuggingFace API key on line 53 in '[RAG.py](https://github.com/Devansh-arora02/Research_Paper_Classifier/blob/main/RAG.py)'
- Run only the '[Generate_Output.py](https://github.com/Devansh-arora02/Research_Paper_Classifier/blob/main/Generate_Output.py)' file.
