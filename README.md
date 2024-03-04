# LLM Interaction System

## Overview
This project implements an agent with an open source language model (LLM) and a memory manager. 
The system is designed to facilitate communication between users and the agent, allowing users to provide instructions, queries, and receive responses from the LLM.

## Approach
The interaction system follows these key steps:
1. Users interact with the `agent` by providing input messages or queries.
2. The `memory manager` stores the new input as a new `Memory`, but before doing so it will check if this `memory` is a duplicate of an older `memory`.
3. The `memory manager` will fetch the most recent memories along with the memories relevant to the user input.
4. The `agent` will use both recent and relevent `memories` to augment the `LLM` prompt.
5. Finally, the `agent` will query the `LLM` and return the response to the user.

## Installation
To set up and run the agent, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yassineelkhadiri/GOODAI.git 
   ```

2. Create a `.env` file in the project directory.
    - Add the following environment variables to the `.env` file:
        ```plaintext
        OPENAI_API_KEY=YOUR_OPENAI_API_KEY
        HUGGING_FACE_API_KEY=YOUR_HUGGING_FACE_API_KEY
        ```

3. Run the following command to install the required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
    - If you prefer using conda for package management:
        - Create a conda environment :
            ```bash
            conda create -n goodai python=3.10
            ```
        - Activate the conda environment:
            ```bash
            conda activate goodai
            ```
        - Activate the conda environment:
            ```bash
            pip install -r requirements.txt
            ```

4. Install ``en_core_web_md``
    ```bash
    python -m spacy download en_core_web_md
    ```

## Usage

To start an interaction with the `agent` or you can simply run, this will use an open source model:
```bash
    python main.py
```

If you wish to use gpt-3.5-turbo, run the following command:
```bash
    python main.py --use-gpt
```
Alternatively, you can refer to the notebooks:
- [Duplicated memory](duplicate.ipynb)
- [Episodic memory](episodic.ipynb)
- [Temporary memory](temporary.ipynb)
- [Contradictory memory](contradictory.ipynb)