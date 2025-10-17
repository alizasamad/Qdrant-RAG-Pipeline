docker run -d \
    -p 9099:9099 \
    --add-host=host.docker.internal:host-gateway \
    -e PIPELINES_URLS="https://github.com/open-webui/pipelines/blob/main/examples/filters/detoxify_filter_pipeline.py" \
    -v pipelines:/app/pipelines \
    --name pipelines \
    --restart always \
    ghcr.io/open-webui/pipelines:main