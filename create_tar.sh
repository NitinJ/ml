#!/bin/bash

OUTPUT="ml.tar.gz"
DOCKERIGNORE=".dockerignore"

echo "📦 Creating project archive: $OUTPUT"

# Check for .dockerignore
if [ ! -f "$DOCKERIGNORE" ]; then
    echo "❌ Error: .dockerignore file not found."
    exit 1
fi

# Check for pv (pipe viewer)
if ! command -v pv &> /dev/null; then
    echo "⚠️ 'pv' not found. Please install it (e.g., 'sudo apt install pv') for progress bar."
    exit 1
fi

# Collect exclude patterns from .dockerignore
EXCLUDE_ARGS=()
while IFS= read -r line; do
    [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
    EXCLUDE_ARGS+=(--exclude="$line")
done < "$DOCKERIGNORE"

# Additional common excludes (optional)
EXCLUDE_ARGS+=(--exclude=".git" --exclude="ml-env.tar")

# Estimate size (approximate, for pv buffer)
EST_SIZE=$(du -sb . | awk '{print $1}')

# Create tar with progress bar
tar -cf - "${EXCLUDE_ARGS[@]}" . \
  | pv -s "$EST_SIZE" \
  | gzip > "$OUTPUT"

echo "✅ Archive created: $OUTPUT"
