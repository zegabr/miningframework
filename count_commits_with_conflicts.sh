
echo "only consider info below if it was run inside mining_resulta3(_?)"
echo "commits with conflicts in csdiff files $(rg --no-ignore --hidden --files-with-matches --regexp '^=======$' . | grep -E '**/csdiff\.py$' | awk -F'/' '{print $4}' | sort -u | wc -l)"
echo "commits with conflicts in diff3 files $(rg --no-ignore --hidden --files-with-matches --regexp '^=======$' . | grep -E '**/diff3\.py$' | awk -F'/' '{print $4}' | sort -u | wc -l)"

