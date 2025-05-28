#!/usr/bin/env bash
#
# rename_capital.sh
# Rename all files in CWD so that the first letter of each filename is uppercase.

shopt -s nullglob
for f in *; do
  # skip non-regular files (directories, symlinks, etc.)
  [ -f "$f" ] || continue

  # split into first character + the rest
  first="${f:0:1}"
  rest="${f:1}"
  # uppercase first char (Bash 4+)
  upfirst="${first^}"
  newname="${upfirst}${rest}"

  # only rename if the name actually changes
  if [[ "$f" != "$newname" ]]; then
    # Use -- to protect against filenames starting with -
    mv -- "$f" "$newname"
  fi
done
