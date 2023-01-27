#!/bin/bash

find . -type d -exec bash -c '
    for d in "$@"; do
        if [ -z "$(find "$d" -mindepth 1 -type d)" ]; then
            if [ ! -e "$d/csdiff.py" ]; then
                rm -r "$d"
            elif [ -e "$d/csdiff.py" -a -e "$d/merge.py" ] ; then

                # correct miningframework mistakes by re-running csdiff
                merge_dir_parent=$(pwd)
                cd "$d"

                    cp ./left.py /home/ze/custom-separators-merge-tool/temp/left.py
                    cp ./right.py /home/ze/custom-separators-merge-tool/temp/right.py
                    cp ./base.py /home/ze/custom-separators-merge-tool/temp/base.py

                    merge_dir=$(pwd)
                    cd /home/ze/custom-separators-merge-tool/
                    printf "running csdiff again for %s\n" "$d"
                    bash /home/ze/custom-separators-merge-tool/csdiff_v3.sh -s "( ) : ," temp/left.py temp/base.py  temp/right.py
                    cd "$merge_dir"

                    cp  /home/ze/custom-separators-merge-tool/temp/csdiff.py ./csdiff.py

                cd "$merge_dir_parent"

                cmp -s "$d/csdiff.py" "$d/diff3.py"
                if [ $? -eq 0 ]; then
                    rm -r "$d"
                fi
            fi
        fi
    done
' bash {} +

find . -type f -name "results.csv" -delete
find . -type f -name "skipped-merge-commits.csv" -delete

find . -type d -empty -delete -not -path "*/\.*"
