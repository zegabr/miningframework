#!/bin/bash
############################################################
# Help                                                     #
############################################################
helpText()
{
   # Display Help
   echo "Syntax: csdiff [-options] MYFILE OLDFILE YOURFILE"
   echo "options:"
   echo "-h                    Print this Help."
   echo "-s \"<separators>\"     Specify the list of separators, e.g. \"{ } ( ) ; ,\""
   echo
}

############################################################
# sed options used                                         #
############################################################
## Descriptions extracted from sed man page: https://linux.die.net/man/1/sed
# -e       - add the script to the commands to be executed
# :a       - Defines a label 'a'
# N        - Append the next line of input into the pattern space.
# $        - Match the last line.
# !        - After the address (or address-range), and before the command, a ! may be inserted, which specifies that the command shall only be executed if the address (or address-range) does not match.
# b[label] - Branch to [label]; if [label] is omitted, branch to end of script.
# s/       - Form: [s/regexp/replacement/] - Attempt to match regexp against the pattern space. If successful, replace that portion matched with replacement.

############################################################
############################################################
#######                   CSDiff                     #######
############################################################
############################################################
############################################################
# Process the input options. Add options as needed.        #
############################################################
while getopts s:h option
do
  case $option in
    h) # display Help
      helpText
      exit 0
      ;;
    s)
      set -f                 # turn off filename expansion
      separators=($OPTARG)   # variable is unquoted
      set +f                 # turn it back on
      ;;
   esac
done

shift $((OPTIND-1))

[ "${1:-}" = "--" ] && shift

############################################################
# Main logic                                               #
############################################################

parameters=("$@")
myFile=${parameters[0]}
oldFile=${parameters[1]}
yourFile=${parameters[2]}
parentFolder=$(echo "${myFile%/*}")
myFileBaseName="$(basename "${myFile}")"
fileExt=$([[ "$myFileBaseName" = *.* ]] && echo ".${myFileBaseName##*.}" || echo '')

myTempFile="${myFile}_temp${fileExt}"
oldTempFile="${oldFile}_temp${fileExt}"
yourTempFile="${yourFile}_temp${fileExt}"

# Dynamically builds the sed command pipeline based on the number of synctatic separators provided
# Build the base substitution script to be passed to the sed command
sedScript=""
for separator in "${separators[@]}"; do
  escapedSeparator=$(printf '%s\n' "$separator" | sed 's/[\[\]\+\.\*\?^$]/\\&/g')
  sedScript+="s/$escapedSeparator/\n\$\$\$\$\$\$\$$escapedSeparator\n\$\$\$\$\$\$\$/g;"
done

# Perform the tokenization of the input file based on the provided separators
sed -e "$sedScript" "$myFile" > "$myTempFile"
sed -e "$sedScript" "$yourFile" > "$yourTempFile"
sed -e "$sedScript" "$oldFile" > "$oldTempFile"

# fix for bug that happens when strings like ======== appears in a multiline comment
# this replace is undone at the end
sed -i 's/=/\$=/g' "$myTempFile"
sed -i 's/=/\$=/g' "$oldTempFile"
sed -i 's/=/\$=/g' "$yourTempFile"


# this is a bash translation of csdiff_python.py of this repo
get_indentation_level() {
    local line=$1
    echo $(expr "$line" : ' *')
}

add_separators_at_indentation_changes() {
    local inputFile="$1"
    awk '
    BEGIN {last_identation_level = 0}
    {
        current_identation_level = length($0) - length(gensub(/^[ \t]+/, "", "g", $0))
        if (current_identation_level != last_identation_level) {
            printf("$$$$$$$\n")
        }
        print
        last_identation_level = current_identation_level
    }
    ' "$inputFile"
}

# run the script to consider identation and override the temporary files again
add_separators_at_indentation_changes "$myTempFile" > myOut
mv myOut "$myTempFile"

add_separators_at_indentation_changes "$oldTempFile" > oldOut
mv oldOut "$oldTempFile"

add_separators_at_indentation_changes "$yourTempFile" > yourOut
mv yourOut "$yourTempFile"

# Runs diff3 against the tokenized inputs, generating a tokenized merged file
midMergedFile="${parentFolder}/mid_merged${fileExt}"
diff3 -m -E "$myTempFile" "$oldTempFile" "$yourTempFile" > $midMergedFile

# Removes the tokenized input files
rm "$myTempFile"
rm "$oldTempFile"
rm "$yourTempFile"

# Removes the tokens from the merged file, generating the final merged file
mergedFile="${parentFolder}/merged${fileExt}"
sed -i ':a;N;$!ba;s/\n\$\$\$\$\$\$\$//g' $midMergedFile

# Renames the tokenized mid merged file
mv "$midMergedFile" "$mergedFile"

# Get the names of left/base/right files
ESCAPED_LEFT=$(printf '%s\n' "${myFile}" | sed -e 's/[\/&]/\\&/g')
ESCAPED_BASE=$(printf '%s\n' "${oldFile}" | sed -e 's/[\/&]/\\&/g')
ESCAPED_RIGHT=$(printf '%s\n' "${yourFile}" | sed -e 's/[\/&]/\\&/g')

ESCAPED_TEMP_LEFT=$(printf '%s\n' "$myTempFile" | sed -e 's/[\/&]/\\&/g')
ESCAPED_TEMP_BASE=$(printf '%s\n' "$oldTempFile" | sed -e 's/[\/&]/\\&/g')
ESCAPED_TEMP_RIGHT=$(printf '%s\n' "$yourTempFile" | sed -e 's/[\/&]/\\&/g')

# Fix the merged file line breaks that got messed up by the CSDiff algorithm.
# TODO: make this universal to other languages, here it will work with python
comment_string="#"
sed -i -e "/^$comment_string/!s/\(<<<<<<< $ESCAPED_TEMP_LEFT\)\(.\+\)/\1\n\2/" $mergedFile
sed -i -e "/^$comment_string/!s/\(<<<<<<< $ESCAPED_TEMP_BASE\)\(.\+\)/\1\n\2/" $mergedFile
sed -i -e "/^$comment_string/!s/\(<<<<<<< $ESCAPED_TEMP_RIGHT\)\(.\+\)/\1\n\2/" $mergedFile
sed -i -e "/^$comment_string/!s/\(||||||| $ESCAPED_TEMP_BASE\)\(.\+\)/\1\n\2/" $mergedFile
sed -i -e "/^$comment_string/!s/\(||||||| $ESCAPED_TEMP_LEFT\)\(.\+\)/\1\n\2/" $mergedFile
sed -i -e "/^$comment_string/!s/\(||||||| $ESCAPED_TEMP_RIGHT\)\(.\+\)/\1\n\2/" $mergedFile
sed -i -e "/^$comment_string/!s/\(>>>>>>> $ESCAPED_TEMP_RIGHT\)\(.\+\)/\1\n\2/" $mergedFile
sed -i -e "/^$comment_string/!s/\(>>>>>>> $ESCAPED_TEMP_LEFT\)\(.\+\)/\1\n\2/" $mergedFile
sed -i -e "/^$comment_string/!s/\(>>>>>>> $ESCAPED_TEMP_BASE\)\(.\+\)/\1\n\2/" $mergedFile
sed -i -e "/^$comment_string/!s/\(=======\)\(.\+\)/\1\n\2/" $mergedFile
sed -i -e "/^$comment_string/!s/$ESCAPED_TEMP_LEFT/$ESCAPED_LEFT/g" $mergedFile
sed -i -e "/^$comment_string/!s/$ESCAPED_TEMP_BASE/$ESCAPED_BASE/g" $mergedFile
sed -i -e "/^$comment_string/!s/$ESCAPED_TEMP_RIGHT/$ESCAPED_RIGHT/g" $mergedFile
sed -i -e "/^$comment_string/!s/=======/\n=======/" $mergedFile
sed -i -e "/^$comment_string/!s/>>>>>>>/\n>>>>>>>/" $mergedFile
sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}' "$mergedFile"
sed -i 's/\$=/=/g' "$mergedFile"

mv "$mergedFile" "${parentFolder}/csdiff${fileExt}"

# Outputs two other files that will be useful for the study: one generated by the diff3 merge
# and another one generated by the 'git merge-file' command, using the diff3.
diff3 -E -m ${myFile} ${oldFile} ${yourFile} > "${parentFolder}/diff3${fileExt}"
git merge-file -p --diff3 ${myFile} ${oldFile} ${yourFile} > "${parentFolder}/git_merge${fileExt}"
