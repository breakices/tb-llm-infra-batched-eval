#!/bin/sh
set -eu
repo=${1:-/workspace/repo}
dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
git -c safe.directory="$repo" -C "$repo" apply --check "$dir/golden/golden.patch"
git -c safe.directory="$repo" -C "$repo" apply "$dir/golden/golden.patch"
