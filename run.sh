#!/usr/bin/env bash
if [ -n "$API_URL" ]; then
  sed -i "s|const baseUrl = \"http://localhost:9090\"|const baseUrl = \"$API_URL\"|g" static/script.js
fi
uvicorn main:app --reload --host 0.0.0.0 --port 9090
