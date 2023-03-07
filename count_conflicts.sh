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
for dir in $(find . -name "csdiff.py" -type f | xargs dirname | sort | uniq); do
  cmp --silent <(grep -v '^\s*$' $dir/csdiff.py) <(grep -v '^\s*$' $dir/diff3.py)
  if [ $? -ne 0 ]; then
    cmp --silent <(grep -v '^\s*$' $dir/csdiff.py) <(grep -v '^\s*$' $dir/merge.py)
    if [ $? -eq 0 ]; then
      files_with_csdiff_different_than_diff3_and_equal_to_merge=$((files_with_csdiff_different_than_diff3_and_equal_to_merge + 1))
    fi
  fi
done

echo "csdiff files with conflicts = " $csdiff_files_with_conflicts
echo "diff3 files with conflicts = " $diff3_files_with_conflicts
echo "csdiff conflicts = " $csdiff_conflicts
echo "diff3 conflicts = " $diff3_conflicts
echo "csdiff files different than diff3 and equal to merge =" $files_with_csdiff_different_than_diff3_and_equal_to_merge