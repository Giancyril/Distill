#!/usr/bin/env bash
git add -A
git commit -m "$1" || echo "Nothing to commit"
