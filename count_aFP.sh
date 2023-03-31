#!/bin/sh

# this scripts can be used to calculate the PP3  and pp4 results for the experiment

has_conflict() {
    if rg -m1 -q '^=======$' "$1"; then
        return 0  # conflict found
    else
        return 1  # no conflicts found
    fi
}

count_file_conflicts() {
    rg -c '^=======$' "$1"
}

count_CaFP() {
    rg -c 'CaFP' "$1"
}

files_are_equal() {
    if diff -q <(sed 's/\s//g' <(tr -d '\n' < "$1")) <(sed 's/\s//g' <(tr -d '\n' < "$2")) >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

count_aFP_on_csdiff() {
    # number of 'CaFP' ocurrences in csdiff.py when diff3.py is equal to merge.py
    total_aFP=0
    total_csdiff_files_with_aFP=0
    for dir in $(find . -name "csdiff.py" -type f | xargs dirname); do
        if files_are_equal "$dir/diff3.py" "$dir/merge.py";  then
          aFP=$(count_CaFP "$dir/csdiff.py") || aFP=0
          total_aFP=$((total_aFP + aFP))
          if [ "$aFP" -gt 0 ]; then
            total_csdiff_files_with_aFP=$((total_csdiff_files_with_aFP + 1))
          fi
        fi
    done
    echo "jonatas csdiff aFP: $total_aFP"
    echo "jonatas csdiff aFP files: $total_csdiff_files_with_aFP"
}

count_aFP_on_diff3() {
    # number of '^=======$' ocurrences in diff3.py when csdiff.py is equal to merge.py
    total_aFP=0
    total_diff3_files_with_aFP=0
    for dir in $(find . -name "diff3.py" -type f | xargs dirname); do
        if files_are_equal "$dir/csdiff.py" "$dir/merge.py";  then
          aFP=$(count_file_conflicts "$dir/diff3.py") || aFP=0
          total_aFP=$((total_aFP + aFP))
          if [ "$aFP" -gt 0 ]; then
            total_diff3_files_with_aFP=$((total_diff3_files_with_aFP + 1))
          fi
        fi
    done
    echo "jonatas diff3 aFP: $total_aFP"
    echo "jonatas diff3 aFP files: $total_diff3_files_with_aFP"
}

scenario_has_csdiff_aFP() {
    # checks if every diff3 is equal to merge and there is at least one csdiff has conflict
    # run this inside merge scenario
    aFP=0
    total_aFP=0
    for dir in $(find . -name "csdiff.py" -type f | xargs dirname); do
        if ! files_are_equal "$dir/diff3.py" "$dir/merge.py"; then
            echo "not a csdiff aFP merge scenario"
            return 1
        fi
        aFP=$(count_CaFP "$dir/csdiff.py") || aFP=0
        total_aFP=$((total_aFP + aFP))
    done
    if [ "$total_aFP" -gt 0 ]; then
        echo "is csdiff aFP merge scenario"
        return 0
    fi
    echo "not a csdiff aFP merge scenario"
    return 1
}

scenario_has_diff3_aFP() {
    # checks if every diff3 is equal to merge and there is at least one csdiff has conflict
    # run this inside merge scenario
    aFP=0
    total_aFP=0
    for dir in $(find . -name "diff3.py" -type f | xargs dirname); do
        if ! files_are_equal "$dir/csdiff.py" "$dir/merge.py"; then
            echo "not a diff3 aFP merge scenario"
            return 1
        fi
        aFP=$(count_file_conflicts "$dir/diff3.py") || aFP=0
        total_aFP=$((total_aFP + aFP))
    done
    if [ "$total_aFP" -gt 0 ]; then
        echo "is diff3 aFP merge scenario"
        return 0
    fi
    echo "not a diff3 aFP merge scenario"
    return 1
}

count_csdiff_possible_aFN() {
 # counts files where diff3 has at least 1 conflict, csdiff has 0 and is different than merge
    possible_aFN_files=0
    for dir in $(find . -name "csdiff.py" -type f | xargs dirname); do
        if ! has_conflict "$dir/csdiff.py"; then
            if has_conflict "$dir/diff3.py"; then
                if ! files_are_equal "$dir/csdiff.py" "$dir/merge.py"; then
                    possible_aFN_files=$((possible_aFN_files + 1))
                fi
            fi
        fi
    done
    echo "possible csdiff aFN files: $possible_aFN_files"
}

count_diff3_possible_aFN() {
 # counts files where csdiff has at least 1 conflict, diff3 has 0 and is different than merge
    possible_aFN_files=0
    for dir in $(find . -name "diff3.py" -type f | xargs dirname); do
        if ! has_conflict "$dir/diff3.py"; then
            if has_conflict "$dir/csdiff.py"; then
                if ! files_are_equal "$dir/diff3.py" "$dir/merge.py"; then
                    possible_aFN_files=$((possible_aFN_files + 1))
                fi
            fi
        fi
    done
    echo "possible diff3 aFN files: $possible_aFN_files"
}

get_project_or_file_data() {
    count_aFP_on_csdiff
    count_aFP_on_diff3
    count_csdiff_possible_aFN
    count_diff3_possible_aFN
    echo
}

get_merge_scenario_data() {
    # run this inside merge scenario
    get_project_or_file_data
    scenario_has_csdiff_aFP
    scenario_has_diff3_aFP
    echo
}