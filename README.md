# 🏙️ **RealEstate Research Tool - RAG - ChatBot**

We are going to build a user-friendly news research tool designed for effortless information retrieval. Users can input article URLs and ask questions to receive relevant insights from the real-estate domain. (But it's features can be extended to any domain.)
![product screenshot](resources/image.png)
### Features

- Load URLs to fetch article content.
- Process article content through LangChain's UnstructuredURL Loader
- Construct an embedding vector using HuggingFace embeddings and leverage ChromaDB as the vectorstore, to enable swift and effective retrieval of relevant information.
- Interact with the LLM's (Llama3 via Groq) by inputting queries and receiving answers along with source URLs.


### Set-up

1. Run the following command to install all dependencies. 

    ```bash
    pip install -r requirements.txt
    ```

2. Create a `.env` file in the project root directory with your API credentials. You can use the `.env.example` file as a template:
    ```bash
    cp .env.example .env
    ```

3. Edit the `.env` file and add your API keys:
    ```text
    # Required: Groq API Key for ChatGroq LLM
    # Get your API key from https://console.groq.com
    GROQ_API_KEY=your_actual_groq_api_key_here

    # Optional: HuggingFace API Token
    # Get your token from https://huggingface.co/settings/tokens
    HF_TOKEN=your_huggingface_token_here
    ```

4. Run the streamlit app by running the following command.

    ```bash
    streamlit run main.py
    ```


### Usage/Examples

The web app will open in your browser after the set-up is complete.

- On the sidebar, you can input URLs directly.

- Initiate the data loading and processing by clicking "Process URLs."

- Observe the system as it performs text splitting, generates embedding vectors using HuggingFace's Embedding Model.

- The embeddings will be stored in ChromaDB.

- One can now ask a question and get the answer based on those news articles

- In the tutorial, we will use the following news articles
  - https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html
  - https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html
  - https://www.cnbc.com/2024/12/17/wall-street-sees-upside-in-2025-for-these-dividend-paying-real-estate-stocks.html


</br>
