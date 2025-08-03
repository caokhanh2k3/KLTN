# sep-dataset

This repository releases the dataset for "Learning to Generate Explainable Stock Predictions using Self-Reflective Large Language Models" [[Paper](https://arxiv.org/abs/2402.03659)].

Note: For text data, only the **raw** dataset is used in our work. The **preprocessed** dataset was used to conduct ablation studies with existing models.

## Dataset Overview

Price and tweet data from 2020 to 2022 of 55 stocks, coming from the top 5 stocks in 11 industries.

The full list of stocks and their companies can be found in **stocktable.pdf**.

## Data Components

This dataset comprises two main components,

- **./tweet**: Tweet data from [Twitter](https://twitter.com/)
- **./price**: Price data from [Yahoo Finance](http://finance.yahoo.com/)

## Data Format

We collect data in the same format as the [Stocknet Dataset](https://github.com/yumoxu/stocknet-dataset).

As the number of tweets have increased exponentially, we also employed a clustering pipeline to obtain the most representative tweets for each day.

### Raw Tweet Data

Format: JSON  
Keys: see [Introduction to Tweet JSON](https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/intro-to-tweet-json)

### Preprocessed Tweet Data

Format: JSON  
Keys: 'text', 'created_at', 'user_id_str'

### Raw Price Data

Format: CSV  
Entries: date, open price, high price, low price, close price, adjusted close price, volume

### Preprocessed Price Data

Format: TXT  
Entries: date, close price, open price, high price, low price, close price change, volume  
Note: _open, high, low, close prices are normalized with the last close price,_ $p\_t = {\tilde{p}\_t / \tilde{p}^c\_{t-1}}-1$.

### Project Overview

<pre>
src/
|-- README.md
|-- requirements.txt
|-- .env
|-- Data
| |-- preprocessed/
| |-- raw/
| |-- tweet/
|
|-- Summarize_and_Prepare_data.ipynb              # Summarizing data and computing technical indicators
| 
| 
|-- Predict_with_tweets/                          # Used for datasets with tweets only
| |-- Exlain.ipynb                                    # Explain and Self-reflect
| |-- Policy.ipynb                                    # Policy-Guided approach
| |-- Policy_ablation.ipynb                           # Policy-Guided ablation study
| |-- SEP_P_ablation.ipynb                            # Phase Predict of the Policy-to-Language Integration into the SEP method (ablation)
| |-- SEP_P.ipynb                                     # Phase Predict of the Policy-to-Language Integration into the SEP method
| |-- SEP.ipynb                                       # Phase Predict of the SEP method
| 
| 
| 
|-- Predict_with_tweets_Tech_Indi/                # Used for datasets with tweets and technical indicators
| |-- Explain_technical_indicator.ipynb               # Explain and Self-reflect
| |-- Policy_technical_indicator.ipynb                # Policy-Guided
| |-- Policy_ablation_technical_indicator.ipynb       # Policy-Guided ablation
| |-- SEP_P_ablation_technical_indicator.ipynb        # Phase Predict of the Policy-to-Language Integration into the SEP method (ablation)
| |-- SEP_P_technical_indicator.ipynb                 # Phase Predict of the Policy-to-Language Integration into the SEP method
| |-- SEP_technical_indicator.ipynb                   # Phase Predict of the SEP method
</pre>

## How to Run

- To save time, intermediate data generated in one phase can be reused in later phases.
- Files that are indented deeper in the structure must be run **after** the files above them.
- Files at the same indentation level (same depth) can be executed in any order.
- Files located in different directories can also be run independently of each other.
- When opening a `.ipynb` notebook, follow the markdown section headers and run the cells from top to bottom in order.
- **Important:** If your GPU does not have enough memory, avoid using "Run all".
  - Large notebooks may cause out-of-memory (OOM) issues due to memory leaks from previous cells.
  - If you encounter OOM, **restart the kernel** and run the remaining steps manually.
  - Each notebook is divided into smaller steps, and intermediate outputs are saved (CSV, JSON, or model checkpoints).
  - Once a step has completed successfully, you can skip rerunning it and proceed with the next related steps.

---

## Files & Directories

### `Summarize_and_Prepare_data.ipynb`

- Purpose: Summarizes data, computes technical indicators, splits training and test sets.
- Outputs: Processed data saved in the `Data/` directory (shared for subsequent steps).

---

### `Predict_with_tweets/`

_Notebooks in this directory work with datasets **without technical indicators**._

- **Explain.ipynb**  
  Generates initial explanations (Explain) and self-reflections (Self-Reflection).  
  Outputs: `..merge..` JSON in the `data` folder and `..comparison..` JSON in the `datasets` folder

  - **SEP.ipynb**  
    Uses the two JSON files to train a model.  
    Output: a CSV file containing predictions.

  - **SEP_P.ipynb**  
    Similar to SEP.ipynb but applies the Policy-to-Language integration approach.  
    Output: a CSV file containing predictions.

    - **SEP_P_ablation.ipynb**  
      Runs after SEP and SEP_P.  
      Uses the previously saved models (in `saved_models/`) to perform an ablation study.

- **Policy.ipynb**  
  Trains a model following the Policy-to-Language approach.

- **Policy_ablation.ipynb**  
  Runs Policy training but skips the Flow Model training phase.

---

### `Predict_with_tweets_Tech_Indi/`

_Notebooks in this directory work with datasets **with tweets and technical indicators**._

- **Explain_technical_indicator.ipynb**  
  Generates initial explanations (Explain) and self-reflections (Self-Reflection).  
  Outputs: `..merge..` JSON in the `data` folder and `..comparison..` JSON in the `datasets` folder

  - **SEP_technical_indicator.ipynb**  
    Uses the two JSON files to train a model.  
    Output: a CSV file containing predictions.

  - **SEP_P_technical_indicator.ipynb**  
    Similar to SEP_technical_indicator.ipynb but applies the Policy-to-Language integration approach.  
    Output: a CSV file containing predictions.

    - **SEP_P_ablation_technical_indicator.ipynb**  
      Runs after SEP_technical_indicator and SEP_P_technical_indicator.  
      Uses the previously saved models (in `saved_models/`) to perform an ablation study.

- **Policy_technical_indicator.ipynb**  
  Trains a model following the Policy-to-Language approach.

- **Policy_ablation_technical_indicator.ipynb**  
  Runs Policy training but skips the Flow Model training phase.
