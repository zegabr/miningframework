#!/bin/bash
#
csdiff_conflicts=0
diff3_conflicts=0
csdiff_files_with_conflicts=0
diff3_files_with_conflicts=0
conflicts=0
files_with_csdiff_different_than_diff3_and_equal_to_merge=0

# csdiff
while IFS= read -r file; do
  conflicts=$(rg -c '^=======$' "$file") || conflicts=0
  csdiff_conflicts=$((csdiff_conflicts + conflicts))
  if [ "$conflicts" -gt 0 ]; then
    csdiff_files_with_conflicts=$((csdiff_files_with_conflicts + 1))
  fi
done < <(rg --files --iglob "**/csdiff.py")

# diff3
while IFS= read -r file; do
  conflicts=$(rg -c '^=======$' "$file") || conflicts=0
  diff3_conflicts=$((diff3_conflicts + conflicts))
  if [ "$conflicts" -gt 0 ]; then
    diff3_files_with_conflicts=$((diff3_files_with_conflicts + 1))
  fi
done < <(rg --files --iglob "**/diff3.py")

# search for folders containing csdiff.py
for dir in $(find . -name "csdiff.py" -type f | xargs dirname); do
    cmp --silent <(sed 's/\s//g' <(tr -d '\n' < $dir/csdiff.py)) <(sed 's/\s//g' <(tr -d '\n' < $dir/merge.py))
    if [ $? -eq 0 ]; then
      files_with_csdiff_different_than_diff3_and_equal_to_merge=$((files_with_csdiff_different_than_diff3_and_equal_to_merge + 1))
    fi
done

echo "csdiff files with conflicts = " $csdiff_files_with_conflicts
echo "diff3 files with conflicts = " $diff3_files_with_conflicts
echo "csdiff conflicts = " $csdiff_conflicts
echo "diff3 conflicts = " $diff3_conflicts
echo "csdiff files different than diff3 and equal to merge =" $files_with_csdiff_different_than_diff3_and_equal_to_merge

echo
# Set the directory to search in (defaults to current directory)
DIR=${1:-.}

# Search for the strings A, B, C, and D and count the number of matches
COUNTaFP=$(rg -c "CaFP" "$DIR" | awk -F ':' '{sum += $2} END {print sum}')
COUNTaFN=$(rg -c "CaFN" "$DIR" | awk -F ':' '{sum += $2} END {print sum}')
COUNTCReduzido=$(rg -c "CReduzido" "$DIR" | awk -F ':' '{sum += $2} END {print sum}')
COUNTCResolvido=$(rg -c "CResolvido" "$DIR" | awk -F ':' '{sum += $2} END {print sum}')
COUNTdiff3FP=$(rg -c "D3FP" "$DIR" | awk -F ':' '{sum += $2} END {print sum}')

# Print the results
echo "below was manually counted"
# echo "csdiff extra conflicts: $COUNTaFP"
# echo "csdiff aFN: $COUNTaFN"
# echo "diff3 FPs: $COUNTdiff3FP"
# echo "conflitos reduzidos: $COUNTCReduzido"
# echo "conflitos resolvidos: $COUNTCResolvido"
# print the result of this equation: -$COUNTaFP - 2*$COUNTaFN + $COUNTCReduzido + 2*$COUNTCResolvido
echo "aumento de produtividade: " $((- 2 * COUNTCResolvido - COUNTCReduzido - 2 * COUNTaFN - COUNTaFP))
