The file `base_asy_full_filter.py` is for retraining Llama on arguments only for the pipeline.

# A guide on reprodcuing the 1rd iteration of the pipeline

- This document describes how to reproduce the **first iteration** of our argument mining pipeline using the provided code components.  
- The pipeline is composed of **two main stages**
---

#### Overview of the Pipeline


##### Step 1: Best performing Llama 3.1 8B on Task 2 (Label Assignment)

**Goal:** Assign zero or more argumentative labels to paragraphs from the test set

1. The best-performing **Llama 3.1 8B** model from *Task 2* is used here
2. This model is then used to run **inference** on all paragraphs from the test set
3. For each paragraph, the model predicts zero or more labels from the predefined inventory of **eight argument categories**.
4. The results are saved as a **CSV file** containing:
   - Each paragraph
   - Its predicted label set
5. This CSV is then processed into a **pickle (.pkl)** file, which stores document-level aggregated information such as:
   - The total count of each label category per document  
   - The frequency distribution of label types  
   - Metadata about paragraph-label mappings  

Examples of these pickle files are available in the directory containing the **MLP** training.

---

##### Step 2: MLP Model for Formalism Classification

**Goal:** Predict whether a document exhibits *formalistic* or *non-formalistic* reasoning.

The **MLP (Multi-Layer Perceptron)** model uses a set of document-level features, including:
- Average argument length
- Total token count per document
- Total number of argumentative labels
- Fraction of each label category relative to the total count

Most features are extracted directly from the **pickle file** produced in Step 1.  
Additional features are computed from the original document and its paragraph structure.

For a detailed description of the MLP’s architecture and feature engineering process, refer to the **MLP module documentation**.

# A guide on reprodcuing the 2nd iteration of the pipeline

- This document describes how to reproduce the **second iteration** of our argument mining pipeline using the provided code components.  
- The pipeline is composed of **three main stages**, each building upon the output of the previous one.

---

#### Overview of the Pipeline

##### Step 1: ModernBERT-Large Fine-Tuned on Task 1 (Argument Detection)

**Goal:** Identify whether each paragraph in the dataset is argumentative.

- All paragraphs are passed through a **ModernBERT-Large** model fine-tuned on *Task 1*.
- The model performs **binary classification**, predicting:
  - `0` → Non-argumentative
  - `1` → Argumentative
- The output is saved as a **CSV file** containing all paragraphs and their corresponding binary labels.

---

##### Step 2: Best performing Llama 3.1 8B on Task 2 (Label Assignment)

**Goal:** Assign zero or more argumentative labels to paragraphs identified as argumentative in Step 1.

1. The best-performing **Llama 3.1 8B** model from *Task 2* is used here
2. This model is then used to run **inference** on the argumentative paragraphs detected in Step 1.
3. For each paragraph, the model predicts zero or more labels from the predefined inventory of **eight argument categories**.
4. The results are saved as a **CSV file** containing:
   - Each paragraph
   - Its predicted label set
5. This CSV is then processed into a **pickle (.pkl)** file, which stores document-level aggregated information such as:
   - The total count of each label category per document  
   - The frequency distribution of label types  
   - Metadata about paragraph-label mappings  

Examples of these pickle files are available in the directory containing the **MLP** training.

---

##### Step 3: MLP Model for Formalism Classification

**Goal:** Predict whether a document exhibits *formalistic* or *non-formalistic* reasoning.

The **MLP (Multi-Layer Perceptron)** model uses a set of document-level features, including:
- Average argument length
- Total token count per document
- Total number of argumentative labels
- Fraction of each label category relative to the total count

Most features are extracted directly from the **pickle file** produced in Step 2.  
Additional features are computed from the original document and its paragraph structure.

For a detailed description of the MLP’s architecture and feature engineering process, refer to the **MLP module documentation**.


# How to run the MLP

* You will have to run the main.py from the terminal with the following command:
```bash 
    python main.py --split --eval_all --inference_choice file_key
```
* #### What does this file key mean?
  * When you have new files in the new experiment they will differ (they should :) ) with the seed values;
  * Source code was organized in a way that you can include the specific key that corresponds to your experiment. That key, will take the exact file. How?

* #### Where to put the data?
  * Data must be in this directory: "data/gold_data/inference_data"
    * Notice that, you will get this data from the dataset link.
  * Make sure you have all 3 seeds;

* #### Where to configure file name so that file_key can get the file that you want?
  * Add the choice to utilities:
    * Go to [utilities.py](utils/utilities.py);
    * Find inference_choice argument in set_parameters() function and add it there:
    ```python
      parser.add_argument('--inference_choice', required=False, type=str, choices=['finetune', 'finetune_filtered', 'X'],
                        help='Different scenarios are evaluated, and that is how we choose it')
    ```
    The given "X" in the above line, which is in the list that is assigned to choices argument, will be your file_key name. It is arbitrary;

* #### Add argument to the filename caller:
  * Go to [process_features.py](../task3/MLP/process_features.py)
  * Find the getfname method of the ProcessFeatures class;
  * Add the same key as an elif conditional (i.e., add the following line by changing X):
  ```python
       elif self.parameters['inference_choice'] == 'X':
            return f'name-of-file-that-you-have-seed-{seed_info}_predicted_label_counts.pkl'
  ```
    
* #### How to see the result?
  * First of all it will be printed out;
  * Secondly, we will save it to the experiment folder of the corresponding experiment number. Currently it is 43 (leave it as it is);
  * To see the result you can go to [exp_43](../task3/MLP/exp_43);
  * Every file name is X_seed_inference_results.csv, where X is the file_key, as you already know :);

**NOTE**: The resulting file will be generated, if and only if it does not exist (saving time by not reproducing the same deterministic computations);
