#!/bin/bash

find . -type d -exec bash -c '
    for d in "$@"; do
        if [ -z "$(find "$d" -mindepth 1 -type d)" ]; then

            if [ ! -e "$d/csdiff.py" ]; then

                # delete folders that does not have csdiff (files with no merge conflict)
                echo "deleting $d as it does not have csdiff.py inside"
                rm -r "$d" &

            elif [ -e "$d/csdiff.py" -a -e "$d/diff3.py" ] ; then

                # delete folders that has csdiff == diff3
                if cmp --silent <(grep -v '^\s*$' $d/csdiff.py) <(grep -v '^\s*$' $d/diff3.py); then
                    echo "deleting $d as it has csdiff == diff3"
                    # rm -r "$d" &
                fi
            fi
        fi
    done
    wait
' bash {} +

find . -type f -name "results.csv" -delete
find . -type f -name "skipped-merge-commits.csv" -delete

echo "deleting empty folders"
find . -type d -empty -delete -not -path "*/\.*"
