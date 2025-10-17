# Introduction
The goal of this project is to determine the optimal RAG pipeline. Preliminary research has already been conducted. 

So far, I have developed a proof-of-concept pipeline to demonstrate that it is possible to connect an external RAG pipeline to Open WebUI. It is imperative that the custom RAG pipeline returns citations as OWUI's default RAG system does. This pipeline is still under development; optimization is currently underway. I expect to perform benchmark testing comparing the two models using [RAGAS](https://docs.ragas.io/en/stable/). 

# Getting Set Up Locally (Docker)
This setup assumes you already have Ollama running locally. OWUI setup may vary slightly if running Ollama as its own Docker container. 

To install dependencies, run `pip install -r requirements`.

## [QDRANT](https://qdrant.tech/documentation/quickstart/)
docker run -p 6333:6333 \
    --name qdrant_secured \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    --env-file .env \
    qdrant/qdrant

## [OWUI](https://docs.openwebui.com/getting-started/quick-start/)
docker run -d \
    -p 3000:8080 \
    --add-host=host.docker.internal:host-gateway \
    -v open-webui:/app/backend/data \
    --env-file .env \
    --name open-webui \
    --restart unless-stopped \
    ghcr.io/open-webui/open-webui:main

**Note**: the .env file contains all environment variables I used in my local setup. I included the .env file for your convenience with default API keys. Please change the API key to something more secure!

## [OWUI Pipelines](https://docs.openwebui.com/features/pipelines/)
docker run -d \
    -p 9099:9099 \
    --add-host=host.docker.internal:host-gateway \
    -e PIPELINES_URLS="https://github.com/open-webui/pipelines/blob/main/examples/filters/detoxify_filter_pipeline.py" \
    -v pipelines:/app/pipelines \
    --name pipelines \
    --restart always \
    ghcr.io/open-webui/pipelines:main

**Note**: check out OWUI docs for more information on adding and connecting to pipelines. 

# Connecting OWUI to custom RAG pipeline
Assuming you have pipelines running as a Docker container, proceed with the following steps:

1. Copy the GitHub Raw URL of the `rag_pipeline.py` file
2. Navigate to the "Pipelines" section in admin settings
3. Paste the URL in the "Install from GitHub URL" section
4. Select the download button and adjust the valves as needed. Don't forget to save!

# Using Qdrant RAG Pipeline
1. In the main chat UI, select Qdrant RAG Pipeline as your model. 
2. Upload files to Qdrant by directly attaching them to OWUI.
3. All set! Query your Qdrant collections at your leisure!

Notice that documents uploaded to Qdrant can be accessed by OWUI at any point. In OWUI's default RAG pipeline, only files uploaded to "Knowledge" are accessible after upload and even then they must be accessed via hashtag. Documents directly uploaded to Qdrant (not through OWUI) are also not accessible by OWUI's default pipeline. With this custom pipeline, you can create your own collection in Qdrant and point the pipeline to that specific collection by adjusting the valves. 

You can also compare the responses returned by OWUI's default RAG vs. the custom RAG pipeline by having two or more models run simultaneously. 

Any and all feedback is appreciated, thanks!

