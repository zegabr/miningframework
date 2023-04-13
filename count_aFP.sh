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

# function to get absolute value of integer
__abs() {
    if [ "$1" -lt 0 ]; then
        echo "$((-$1))"
    else
        echo "$1"
    fi
}


# function that given a directory, count the sum of the __abs (base - left) and (base - right) and (left - right) where each variable is the number of lines of a file
sum_of_line_diffs() {
    echo "sum of line diff between base left and right:"
    total_total=0
    for dir in $(find . -name "csdiff.py" -type f | xargs dirname); do
        total=0
        base=$(wc -l < "$dir/base.py")
        left=$(wc -l < "$dir/left.py")
        right=$(wc -l < "$dir/right.py")
        # tota = |base - left| + |base - right| + |left - right|
        # if total is more than 0, echo the dir and the local total
        total=$((total + $(__abs $((base - left))) + $(__abs $((base - right)))))
        if files_are_equal "$dir/diff3.py" "$dir/merge.py";  then
            aFP=$(count_file_conflicts "$dir/csdiff.py") || aFP=0
            total_total=$((total_total + total))
            if [ "$aFP" -gt 0 ]; then
                echo "(aFPyes)$dir $total "
            else
                echo "(aFPno)$dir $total "
            fi
        else
            echo "(aFPno)$dir $total "
        fi
    done
    echo "total: $total_total"
}

# function that do the same as sum_of_line_diffs but counting the numer of conflicting blocks between base/left and base/right
sum_of_diff_blocks() {
    echo "sum of conflict diff between base left and right:"
    total_total=0
    for dir in $(find . -name "csdiff.py" -type f | xargs dirname); do
        left=$(diff "$dir/base.py" "$dir/left.py" | grep -c '^[0-9]')
        right=$(diff "$dir/base.py" "$dir/right.py" | grep -c '^[0-9]')
        if files_are_equal "$dir/diff3.py" "$dir/merge.py";  then
            aFP=$(count_file_conflicts "$dir/csdiff.py") || aFP=0
            if [ "$aFP" -gt 0 ]; then
                total=$(diff3 -m -A "$dir/left.py" "$dir/base.py" "$dir/right.py" | grep -c '^=======')
                total_total=$((total_total + total))
                echo "(aFPyes)$dir $total "
            else
                echo "(aFPno)$dir $total "
            fi
        else
            echo "(aFPno)$dir $total "
        fi
    done
    echo "total: $total_total"
}


count_aFP_on_csdiff() {
    # number of '^=======$' ocurrences in csdiff.py when diff3.py is equal to merge.py
    total_aFP=0
    total_csdiff_files_with_aFP=0
    for dir in $(find . -name "csdiff.py" -type f | xargs dirname); do
        if files_are_equal "$dir/diff3.py" "$dir/merge.py";  then
          aFP=$(count_file_conflicts "$dir/csdiff.py") || aFP=0
          total_aFP=$((total_aFP + aFP))
          if [ "$aFP" -gt 0 ]; then
            # echo "$dir"
            total_csdiff_files_with_aFP=$((total_csdiff_files_with_aFP + 1))
          fi
        fi
    done
    # echo "jonatas csdiff aFP: $total_aFP"
    echo "jonatas csdiff aFP files: $total_csdiff_files_with_aFP"
    echo -n "$total_csdiff_files_with_aFP," >> ~/Desktop/a.csv
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
    # echo "jonatas diff3 aFP: $total_aFP"
    echo "jonatas diff3 aFP files: $total_diff3_files_with_aFP"
    echo -n "$total_diff3_files_with_aFP," >> ~/Desktop/a.csv
}

scenario_has_csdiff_aFP() {
    # checks if every diff3 is equal to merge and there is at least one csdiff has conflict
    # run this inside merge scenario
    aFP=0
    total_aFP=0
    for dir in $(find . -name "csdiff.py" -type f | xargs dirname); do
        if ! files_are_equal "$dir/diff3.py" "$dir/merge.py"; then
            echo "not a csdiff aFP merge scenario"
            echo -n "FALSO," >> ~/Desktop/a.csv
            return 0
        fi
        aFP=$(count_file_conflicts "$dir/csdiff.py") || aFP=0
        total_aFP=$((total_aFP + aFP))
    done
    if [ "$total_aFP" -gt 0 ]; then
        echo "is csdiff aFP merge scenario"
        echo -n "VERDADEIRO," >> ~/Desktop/a.csv
        return 0
    fi
    echo "not a csdiff aFP merge scenario"
    echo -n "FALSO," >> ~/Desktop/a.csv
    return 0
}

scenario_has_diff3_aFP() {
    # checks if every diff3 is equal to merge and there is at least one csdiff has conflict
    # run this inside merge scenario
    aFP=0
    total_aFP=0
    for dir in $(find . -name "diff3.py" -type f | xargs dirname); do
        if ! files_are_equal "$dir/csdiff.py" "$dir/merge.py"; then
            echo "not a diff3 aFP merge scenario"
            echo -n "FALSO," >> ~/Desktop/a.csv
            return 0
        fi
        aFP=$(count_file_conflicts "$dir/diff3.py") || aFP=0
        total_aFP=$((total_aFP + aFP))
    done
    if [ "$total_aFP" -gt 0 ]; then
        echo "is diff3 aFP merge scenario"
        echo -n "VERDADEIRO," >> ~/Desktop/a.csv
        return 0
    fi
    echo "not a diff3 aFP merge scenario"
    echo -n "FALSO," >> ~/Desktop/a.csv
    return 0
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
    echo -n "$possible_aFN_files," >> ~/Desktop/a.csv
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
    # count_diff3_possible_aFN
    echo
}

get_merge_scenario_data() {
    # run this inside project, where we have the merge scenarios as result of ls command
    arr=($(ls -1)) # merge scenarios
    echo "" > ~/Desktop/a.csv
    for elem in "${arr[@]}"; do
        echo "" >> ~/Desktop/a.csv
        echo "$elem"
        echo -n "$elem," >> ~/Desktop/a.csv
        cd $elem
        count_conflicts
        get_project_or_file_data
        scenario_has_csdiff_aFP
        scenario_has_diff3_aFP
        cd ..
    done
    echo "cat ~/Desktop/a.csv"
    cat ~/Desktop/a.csv
}
