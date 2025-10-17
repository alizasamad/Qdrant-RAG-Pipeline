docker run -p 6333:6333 \
    --name qdrant_secured \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    --env-file .env \
    qdrant/qdrant