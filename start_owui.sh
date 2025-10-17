docker run -d \
    -p 3000:8080 \
    --add-host=host.docker.internal:host-gateway \
    -v open-webui:/app/backend/data \
    --env-file .env \
    --name open-webui \
    --restart unless-stopped \
    ghcr.io/open-webui/open-webui:main