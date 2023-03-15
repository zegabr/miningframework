
# this script has to be run inside the results_path defined in test.sh
echo "commits with conflicts in csdiff files $(rg --no-ignore --hidden --files-with-matches --regexp '^=======$' . | grep -E '**/csdiff\.py$' | awk -F'/' '{print $4}' | sort -u | wc -l)"
echo "commits with conflicts in diff3 files $(rg --no-ignore --hidden --files-with-matches --regexp '^=======$' . | grep -E '**/diff3\.py$' | awk -F'/' '{print $4}' | sort -u | wc -l)"
