#!/bin/bash

find . -type d -exec bash -c '
    for d in "$@"; do
        if [ -z "$(find "$d" -mindepth 1 -type d)" ]; then
            if [ -e "$d/csdiff.py" -a -e "$d/diff3.py" ] ; then
                cmp -s "$d/csdiff.py" "$d/diff3.py"
                if [ $? -eq 1 ]; then
                    # re-run csdiff in the folder
                    printf "running csdiff again for %s\n" "$d"
                    bash /home/ze/miningframework/dependencies/csdiff_v3.sh -s "( ) : ," "$d"/left.py "$d"/base.py "$d"/right.py
                fi
            fi
        fi
    done
' bash {} +

