find . -type d -exec bash -c '
    for d in "$@"; do
        if [ -z "$(find "$d" -mindepth 1 -type d)" ]; then
            if [ ! -e "$d/csdiff.py" ]; then
                rm -r "$d"
            fi
        fi
    done
' bash {} +

find . -type f -name "result.csv" -delete

find . -type d -empty -delete -not -path "*/\.*"
