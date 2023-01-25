find . -type d -exec bash -c '
    for d in "$@"; do
        if [ -z "$(find "$d" -mindepth 1 -type d)" ]; then
            if [ ! -e "$d/csdiff.py" ]; then
                rm -r "$d"
            elif [ -e "$d/csdiff.py" -a -e "$d/merge.py" ] ; then
                cmp -s "$d/csdiff.py" "$d/merge.py"
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
