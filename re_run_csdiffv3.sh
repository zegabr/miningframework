#!/bin/bash

find . -type d -exec bash -c '
    for d in "$@"; do
        if [ -z "$(find "$d" -mindepth 1 -type d)" ]; then
            if [ -e "$d/csdiff.py" -a -e "$d/diff3.py" ] ; then

                # since we are creating folders, this pass file will mark the created ones
                if [ ! -e "$d/pass" ]; then

                    # re-run csdiff in a copy of the folder
                    cp -r "$d" "$d"_after_re_run
                    merge_dir_parent=$(pwd)
                    cd "$d"_after_re_run
                        touch pass
                        printf "running csdiff again for %s\n" "$d"
                        bash /home/ze/miningframework/dependencies/csdiff_v3.sh -s "( ) : ," ./left.py ./base.py ./right.py
                    cd "$merge_dir_parent"
                fi
            fi
        fi
    done
' bash {} +

